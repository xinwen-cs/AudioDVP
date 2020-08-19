#include <vector>
#include <torch/extension.h>


// CUDA forward declarations
std::vector<torch::Tensor> rasterize_triangles_cuda_forward(
    const torch::Tensor &vertices,
    const torch::Tensor &triangles,
    const int image_width,
    const int image_height,
    torch::Tensor &px_triangle_ids,
    torch::Tensor &px_barycentric_coordinates,
    torch::Tensor &z_buffer
);


// CUDA backward declarations
torch::Tensor rasterize_triangles_cuda_backward(
    const torch::Tensor &vertices,
    const torch::Tensor &triangles,
    const torch::Tensor &triangle_ids,
    const torch::Tensor &barycentric_coordinates,
    const torch::Tensor &df_dbarycentric_coordinates,
    const int image_width,
    const int image_height,
    torch::Tensor &df_dvertices
);


// C++ interface
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


std::vector<torch::Tensor> rasterize_triangles_forward(
    const torch::Tensor &vertices,
    const torch::Tensor &triangles,
    const int image_width,
    const int image_height,
    torch::Tensor &px_triangle_ids,
    torch::Tensor &px_barycentric_coordinates,
    torch::Tensor &z_buffer
)
{
    CHECK_INPUT(vertices);
    CHECK_INPUT(triangles);
    CHECK_INPUT(px_triangle_ids);
    CHECK_INPUT(px_barycentric_coordinates);
    CHECK_INPUT(z_buffer);

    return rasterize_triangles_cuda_forward(
                vertices,
                triangles,
                image_width,
                image_height,
                px_triangle_ids,
                px_barycentric_coordinates,
                z_buffer);
}


torch::Tensor rasterize_triangles_backward(
    const torch::Tensor &vertices,
    const torch::Tensor &triangles,
    const torch::Tensor &triangle_ids,
    const torch::Tensor &barycentric_coordinates,
    const torch::Tensor &df_dbarycentric_coordinates,
    const int image_width,
    const int image_height,
    torch::Tensor &df_dvertices
)
{
    CHECK_INPUT(vertices);
    CHECK_INPUT(triangles);
    CHECK_INPUT(triangle_ids);
    CHECK_INPUT(barycentric_coordinates);
    CHECK_INPUT(df_dbarycentric_coordinates);
    CHECK_INPUT(df_dvertices);

    return rasterize_triangles_cuda_backward(
                vertices,
                triangles,
                triangle_ids,
                barycentric_coordinates,
                df_dbarycentric_coordinates,
                image_width,
                image_height,
                df_dvertices
    );
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &rasterize_triangles_forward, "Rasterize forward (CUDA)");
    m.def("backward", &rasterize_triangles_backward, "Rasterize backward (CUDA)");
}
