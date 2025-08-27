# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# NOTE: this file may be removed once we move to a dynamo frontend

import functools
from typing import Any, Callable, Optional, Union

import torch
import torch.utils._pytree as pytree
from torch._higher_order_ops.utils import (
    clone_outputs_aliasing_inputs,
    save_tensors_and_symints_for_backward,
    saved_tensors_and_symints,
)
from torch._ops import HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx import GraphModule
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode, track_tensor_tree


# Proxy the HOP instead of inlining into it
_DEFER_INLINING = False


class LocalMapHOP(HigherOrderOperator):
    def __init__(self) -> None:
        super().__init__("local_map_hop")

    def __call__(self, fw_gm: GraphModule, *args: Any, **kwargs: Any) -> Any:
        return super().__call__(fw_gm, *args, **kwargs)


local_map_hop = LocalMapHOP()


class LocalMapBackwardHOP(HigherOrderOperator):
    def __init__(self) -> None:
        super().__init__("local_map_hop_backward")

    def __call__(self, bw_gm: GraphModule, *args: Any, **kwargs: Any) -> Any:
        return super().__call__(bw_gm, *args, **kwargs)


local_map_hop_backward = LocalMapBackwardHOP()


def create_hop_fw_bw(
    fw_gm: GraphModule,
    *_args: Any,
) -> tuple[GraphModule, GraphModule, int]:
    """
    Traces a joint, applies passes and partitions it
    """
    # Keeping these imports here
    # Avoid circular dependencies once we upstream with dynamo frontend
    from torch._dispatch.python import suspend_functionalization
    from torch._functorch.aot_autograd import AOTConfig, create_joint
    from torch._guards import detect_fake_mode
    from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
    from torch._subclasses.functional_tensor import disable_functional_mode
    from torch.fx.experimental.proxy_tensor import disable_proxy_modes_tracing, make_fx

    dummy_aot_config = AOTConfig(
        fw_compiler=None,  # type: ignore[arg-type]
        bw_compiler=None,  # type: ignore[arg-type]
        partition_fn=None,  # type: ignore[arg-type]
        decompositions={},
        num_params_buffers=0,
        aot_id=0,
        keep_inference_input_mutations=False,
    )

    with suspend_functionalization(), disable_functional_mode():
        with disable_proxy_modes_tracing():
            # create a tensor (fake) from a compiler wrapped FunctionalTensor
            def _from_fun(t: Any) -> Any:
                if isinstance(t, torch.Tensor):
                    return torch.empty_strided(
                        t.size(),
                        t.stride(),
                        device=t.device,
                        dtype=t.dtype,
                        requires_grad=t.requires_grad,
                    )
                return t

            # If someone runs this hop under the default compiler backend ("eager")
            # Then this path will be run with the actual user inputs. We convert them
            # to fake tensors in order to not perform any actual compute.

            fake_mode = detect_fake_mode(_args)
            if fake_mode is None:
                fake_mode = FakeTensorMode(allow_non_fake_inputs=True)

            with fake_mode:
                fw_inputs = pytree.tree_map(_from_fun, _args)

            assert all(
                isinstance(t, (FakeTensor, int, torch.SymInt)) for t in fw_inputs
            ), f"Unexpected element in {fw_inputs=}"

            example_grads = pytree.tree_map(
                _from_fun,
                fw_gm(*fw_inputs),
            )
            if not isinstance(example_grads, (list, tuple)):
                example_grads = [example_grads]

            num_fw_inputs = len(fw_inputs)
            num_fw_outputs = len(example_grads)

        def joint_f(
            *primals_and_tangents: list[torch.Tensor],
        ) -> Any:
            primals = primals_and_tangents[:num_fw_inputs]
            tangents = primals_and_tangents[num_fw_inputs:]

            optional_grads = []
            for example_grad in example_grads:
                if example_grad.requires_grad:
                    optional_grads.append(example_grad)

            def prepare_fw_with_masks(fn: Callable[..., Any]) -> Callable[..., Any]:
                def fw_with_masks(*args: Any) -> tuple[tuple[Any], list[bool]]:
                    fw_out = fn(*args)
                    assert isinstance(fw_out, tuple), (
                        "Dynamo traced submodule should return tuple"
                    )
                    return fw_out, [
                        True
                        if isinstance(ret, torch.Tensor) and ret.requires_grad
                        else False
                        for ret in fw_out
                    ]

                return fw_with_masks

            fw_outs, grads = create_joint(
                prepare_fw_with_masks(fw_gm), aot_config=dummy_aot_config
            )(primals, tangents)

            maybe_clone = clone_outputs_aliasing_inputs(primals_and_tangents)
            # put grads first to work with existing hop utils
            return pytree.tree_map(maybe_clone, (*grads, *fw_outs))

        primals_and_tangents = [*fw_inputs, *example_grads]
        joint_hop_gm = make_fx(joint_f)(*primals_and_tangents)

        from torch._functorch._aot_autograd.graph_compile import prepare_for_partitioner
        from torch._inductor.compile_fx import partition_fn

        # Match partitioner convention
        prepped_joint_hop_gm = prepare_for_partitioner(
            joint_hop_gm, num_fw_inputs, num_fw_outputs
        )
        # Also runs joint passes
        new_fw_gm, new_bw_gm = partition_fn(
            prepped_joint_hop_gm,
            [],
            num_fwd_outputs=num_fw_outputs,
            static_lifetime_input_indices=[],
        )

        # Propagate meta onto fw/bw graphs, later will be set on proxied nodes
        local_map_kwargs = fw_gm.meta["local_map_kwargs"]  # type: ignore[attr-defined]

        new_fw_gm.meta["local_map_kwargs"] = local_map_kwargs
        new_bw_gm.meta["local_map_kwargs"] = local_map_kwargs
        # Okay because Autoparallel assumes same sharding between param and grads
        new_bw_gm.meta["local_map_kwargs"]["in_placements"] = local_map_kwargs[
            "out_placements"
        ]
        new_bw_gm.meta["local_map_kwargs"]["out_placements"] = local_map_kwargs[
            "in_placements"
        ]

        return new_fw_gm, new_bw_gm, num_fw_outputs


class LocalMapAutogradOp(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        fw_gm: GraphModule,
        bw_gm: GraphModule,
        num_fw_outs: int,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[Optional[torch.Tensor], ...]:
        ctx.bw_gm = bw_gm

        with torch._C._AutoDispatchBelowAutograd():
            fw_outs_with_saved_activations = local_map_hop(fw_gm, *args, **kwargs)

        fw_outs = fw_outs_with_saved_activations[:num_fw_outs]
        saved_activations = fw_outs_with_saved_activations[num_fw_outs:]
        save_tensors_and_symints_for_backward(ctx, saved_activations)

        return fw_outs

    @staticmethod
    def backward(
        ctx: Any, *grads: tuple[torch.Tensor]
    ) -> tuple[Optional[torch.Tensor], ...]:
        saved_activations = saved_tensors_and_symints(ctx)
        with torch._C._AutoDispatchBelowAutograd():
            grad_ins = local_map_hop_backward(ctx.bw_gm, *saved_activations, *grads)
        return None, None, None, *grad_ins


@local_map_hop.py_impl(torch._C.DispatchKey.Autograd)
def autograd_key(
    fw_gm: GraphModule,
    *args: Any,
    **kwargs: Any,
) -> Any:
    if _DEFER_INLINING:
        fw_gm, bw_gm, num_fw_outs = create_hop_fw_bw(fw_gm, *args)
        return LocalMapAutogradOp.apply(fw_gm, bw_gm, num_fw_outs, *args, **kwargs)

    return fw_gm(*args, **kwargs)


@local_map_hop.py_functionalize_impl
def functional_mode_key(
    ctx: Any, fw_gm: GraphModule, *args: Any, **kwargs: Any
) -> tuple[torch.Tensor]:
    assert not kwargs

    unwrapped_inputs = ctx.unwrap_tensors(args)
    with ctx.redispatch_to_next():
        out = local_map_hop(fw_gm, *unwrapped_inputs)
        return ctx.wrap_tensors(out)


@local_map_hop.py_impl(FakeTensorMode)
def fake_mode_key(
    mode: FakeTensorMode,
    fw_gm: GraphModule,
    *args: Any,
    **kwargs: Any,
) -> tuple[torch.Tensor]:
    with mode:
        return fw_gm(*args, **kwargs)


def proxy_mode_key_common(
    hop: Union[LocalMapHOP, LocalMapBackwardHOP],
    call_hop: Callable[..., Any],
    proxy_mode: ProxyTorchDispatchMode,
    gm: GraphModule,
    *args: Any,
    **kwargs: Any,
) -> tuple[torch.Tensor]:
    assert proxy_mode is not None, (
        "Mode should always be enabled for python fallback key"
    )
    assert len(kwargs) == 0

    example_out = hop(gm, *args, **kwargs)
    proxy_args = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, args)  # type: ignore[union-attr]

    out_proxy = proxy_mode.tracer.create_proxy(
        "call_function", call_hop, proxy_args, {}
    )

    # extract local_map args, post-dispatch operates on GraphModules
    assert gm.meta["local_map_kwargs"]
    local_map_kwargs = gm.meta["local_map_kwargs"]

    # propagate local_map args to the call_function node
    out_proxy.node.meta["local_map_kwargs"] = local_map_kwargs
    return track_tensor_tree(
        example_out, out_proxy, constant=None, tracer=proxy_mode.tracer
    )


@local_map_hop.py_impl(ProxyTorchDispatchMode)
def proxy_mode_key(
    proxy_mode: ProxyTorchDispatchMode,
    fw_gm: GraphModule,
    *args: Any,
    **kwargs: Any,
) -> tuple[torch.Tensor]:
    # TODO: get rid of this when we can install as a subgraph
    def call_local_map(*_args: Any, **_kwargs: Any) -> Any:
        return functools.partial(local_map_hop, fw_gm)(*_args, **_kwargs)

    return proxy_mode_key_common(
        local_map_hop, call_local_map, proxy_mode, fw_gm, *args, **kwargs
    )


@local_map_hop_backward.py_impl(torch._C.DispatchKey.Autograd)
def bw_autograd_key(
    bw_gm: GraphModule,
    *args: Any,
    **kwargs: Any,
) -> tuple[torch.Tensor]:
    assert not _DEFER_INLINING, "Not supported yet"
    # NOTE: no double backward support
    return bw_gm(*args, **kwargs)


@local_map_hop_backward.py_functionalize_impl
def bw_functional_mode_key(
    ctx: Any, bw_gm: GraphModule, *args: Any, **kwargs: Any
) -> tuple[torch.Tensor]:
    assert not kwargs

    unwrapped_inputs = ctx.unwrap_tensors(args)
    with ctx.redispatch_to_next():
        out = local_map_hop_backward(bw_gm, *unwrapped_inputs)
        return ctx.wrap_tensors(out)


@local_map_hop_backward.py_impl(FakeTensorMode)
def bw_fake_mode_key(
    mode: FakeTensorMode,
    bw_gm: GraphModule,
    *args: Any,
    **kwargs: Any,
) -> tuple[torch.Tensor]:
    with mode:
        return bw_gm(*args, **kwargs)


@local_map_hop_backward.py_impl(ProxyTorchDispatchMode)
def bw_proxy_mode_key(
    proxy_mode: ProxyTorchDispatchMode,
    bw_gm: GraphModule,
    *args: Any,
    **kwargs: Any,
) -> tuple[torch.Tensor]:
    # TODO: get rid of this when we can install as a subgraph
    def call_local_map_backward(*_args: Any, **_kwargs: Any) -> Any:
        return functools.partial(local_map_hop_backward, bw_gm)(*_args, **_kwargs)

    return proxy_mode_key_common(
        local_map_hop_backward,
        call_local_map_backward,
        proxy_mode,
        bw_gm,
        *args,
        **kwargs,
    )


# Running HOP in eager with real tensors
@local_map_hop.py_impl(torch._C.DispatchKey.CPU)
@local_map_hop.py_impl(torch._C.DispatchKey.CUDA)
def real_impl(
    fw_gm: GraphModule,
    *args: Any,
    **kwargs: Any,
) -> tuple[torch.Tensor]:
    return fw_gm(*args, **kwargs)


# Running HOP in eager with real tensors
@local_map_hop_backward.py_impl(torch._C.DispatchKey.CPU)
@local_map_hop_backward.py_impl(torch._C.DispatchKey.CUDA)
def bw_real_impl(
    bw_gm: GraphModule,
    *args: Any,
    **kwargs: Any,
) -> tuple[torch.Tensor]:
    return bw_gm(*args, **kwargs)
