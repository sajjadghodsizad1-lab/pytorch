# Owner(s): ["module: PrivateUse1"]

import torch
import torch_openreg  # noqa: F401
from torch.testing._internal.common_utils import (
    run_tests,
    skipIfTorchDynamo,
    skipIfXpu,
    TestCase,
)


class TestOperators(TestCase):
    @skipIfTorchDynamo()
    def test_operator_registration(self):
        self.assertTrue(
            torch._C._dispatch_has_kernel_for_dispatch_key(
                "aten::empty.memory_format", torch.DispatchKey.PrivateUse1
            )
        )

    def test_factory(self):
        x = torch.empty(3, device="openreg")
        self.assertEqual(x.device.type, "openreg")
        self.assertEqual(x.shape, torch.Size([3]))

        y = torch.zeros(3, device="openreg")
        self.assertEqual(y.device.type, "openreg")
        self.assertEqual(y.shape, torch.Size([3]))

        z = torch.tensor((), device="openreg")
        self.assertEqual(z.device.type, "openreg")
        self.assertEqual(z.shape, torch.Size([0]))

    def test_fake_tensor(self):
        with torch._subclasses.fake_tensor.FakeTensorMode():
            a = torch.empty(1, device="openreg")
            b = torch.empty(1, device="openreg:0")
            result = a + b  # noqa: F841

    def test_named_tensor(self):
        return torch.empty([2, 3, 4, 5], device="openreg", names=["N", "C", "H", "W"])

    def test_printing(self):
        a = torch.ones(20, device="openreg")
        print(a)

    def test_data_dependent_output(self):
        cpu_a = torch.randn(10)
        a = cpu_a.to(device="openreg")
        mask = a.gt(0)
        out = torch.masked_select(a, mask)

        self.assertEqual(out, cpu_a.masked_select(cpu_a.gt(0)))

    def test_expand(self):
        x = torch.tensor([[1], [2], [3]], device="openreg")
        y = x.expand(3, 2)
        self.assertEqual(y.to(device="cpu"), torch.tensor([[1, 1], [2, 2], [3, 3]]))
        self.assertEqual(x.data_ptr(), y.data_ptr())

    def test_resize(self):
        tensor_cpu = torch.randn([4, 4])

        tensor_openreg = tensor_cpu.openreg()
        self.assertTrue(tensor_openreg.size() == torch.Size([4, 4]))

        storage_openreg = tensor_openreg.storage()
        self.assertTrue(storage_openreg.size() == 16)

        tensor_openreg.resize_(2, 2, 2, 2)
        self.assertTrue(tensor_openreg.size() == torch.Size([2, 2, 2, 2]))

        storage_openreg = tensor_openreg.storage()
        self.assertTrue(storage_openreg.size() == 16)


class TestSTUB(TestCase):
    def test_backend_dispatchstub(self):
        x_cpu = torch.randn(2, 2, 3, dtype=torch.float32, device="cpu")
        x_openreg = x_cpu.to("openreg")

        y_cpu = torch.abs(x_cpu)
        y_openreg = torch.abs(x_openreg)
        self.assertEqual(y_cpu, y_openreg.cpu())

        o_cpu = torch.randn(2, 2, 6, dtype=torch.float32, device="cpu")
        o_openreg = o_cpu.to("openreg")
        # output operand with resize flag is False in TensorIterator.
        torch.abs(x_cpu, out=o_cpu[:, :, 0:6:2])
        torch.abs(x_openreg, out=o_openreg[:, :, 0:6:2])
        self.assertEqual(o_cpu, o_openreg.cpu())

        # output operand with resize flag is True in TensorIterator and
        # convert output to contiguous tensor in TensorIterator.
        torch.abs(x_cpu, out=o_cpu[:, :, 0:6:3])
        torch.abs(x_openreg, out=o_openreg[:, :, 0:6:3])
        self.assertEqual(o_cpu, o_openreg.cpu())


class TestQuantization(TestCase):
    @skipIfXpu(msg="missing kernel for openreg")
    def test_quantize(self):
        x = torch.randn(3, 4, 5, dtype=torch.float32, device="openreg")
        quantized_tensor = torch.quantize_per_tensor(x, 0.1, 10, torch.qint8)
        self.assertEqual(quantized_tensor.device, torch.device("openreg:0"))
        self.assertEqual(quantized_tensor.dtype, torch.qint8)


class TestAutogradFunction(TestCase):
    def test_compile_autograd_function_returns_self(self):
        in_ref = torch.randn(4, device="openreg", requires_grad=True)
        out_ref = torch.ops.openreg.custom_autograd_fn_returns_self(in_ref)
        out_ref.sum().backward()

        in_test = in_ref.detach().clone().requires_grad_(True)
        # TODO(FFFrog): Need to support inductor for OpenReg first.
        out_test = torch.compile(backend="aot_eager")(
            torch.ops.openreg.custom_autograd_fn_returns_self
        )(in_test)
        out_test.sum().backward()

        self.assertEqual(out_ref, out_test)
        self.assertEqual(in_ref.grad, in_test.grad)

    @skipIfTorchDynamo("Temporary disabled due to torch._ops.OpOverloadPacket")
    def test_compile_autograd_function_aliasing(self):
        in_ref = torch.randn(4, device="openreg", requires_grad=True)
        out_ref = torch.ops.openreg.custom_autograd_fn_aliasing(in_ref)
        out_ref.sum().backward()

        in_test = in_ref.detach().clone().requires_grad_(True)
        # TODO(FFFrog): Need to support inductor for OpenReg first.
        out_test = torch.compile(backend="aot_eager")(
            torch.ops.openreg.custom_autograd_fn_aliasing
        )(in_test)
        out_test.sum().backward()

        self.assertEqual(out_ref, out_test)
        self.assertEqual(in_ref.grad, in_test.grad)


class TestFallback(TestCase):
    def test_scalar_type_fallback(self):
        x_cpu = torch.Tensor([[0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2]]).to(torch.int64)
        x = torch.triu_indices(3, 3, device="openreg")
        self.assertEqual(x_cpu, x)

    def test_tensor_type_fallback(self):
        x = torch.Tensor([[1, 2, 3], [2, 3, 4]]).to("openreg")
        y = torch.Tensor([1, 0, 2]).to("openreg")
        self.assertTrue(x.device.type, "openreg")
        self.assertFalse(x.is_cpu)

        z_cpu = torch.Tensor([[0, 2, 1], [1, 3, 2]])
        # call sub op, which will fallback to cpu
        z = torch.sub(x, y)
        self.assertEqual(z_cpu, z)

        # call index op, which will fallback to cpu
        z_cpu = torch.Tensor([3, 1])
        y = torch.Tensor([1, 0]).long().to("openreg")
        z = x[y, y]
        self.assertEqual(z_cpu, z)

    def test_tensorlist_type_fallback(self):
        # create tensors located in custom device
        v_openreg = torch.Tensor([1, 2, 3]).to("openreg")
        # create result tensor located in cpu
        z_cpu = torch.Tensor([2, 4, 6])
        # create tensorlist for foreach_add op
        x = (v_openreg, v_openreg)
        y = (v_openreg, v_openreg)

        # Check that our device is correct.
        self.assertTrue(v_openreg.device.type == "openreg")
        self.assertFalse(v_openreg.is_cpu)

        # call _foreach_add op, which will fallback to cpu
        z = torch._foreach_add(x, y)
        self.assertEqual(z_cpu, z[0])
        self.assertEqual(z_cpu, z[1])


if __name__ == "__main__":
    run_tests()
