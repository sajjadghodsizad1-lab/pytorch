#include <ATen/native/DispatchStub.h>
#include <ATen/ATen.h>


namespace at::native {

using mm_complex_fn = at::Tensor& (*)(
  const at::Tensor&,
  const at::Tensor&,
  Tensor&
);

DECLARE_DISPATCH(mm_complex_fn, mm_complex_stub);

}