/*
Following https://github.com/google/tf_mesh_renderer/blob/master/mesh_renderer/kernels/rasterize_triangles_impl.cc
*/

#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>


__device__ static float atomicMin(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do
    {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed, __float_as_int(::fminf(val, __int_as_float(assumed))));
    }
    while (assumed != old);

    return __int_as_float(old);
}


__device__ constexpr float kDegenerateBarycentricCoordinatesCutoff() { return 0.9f; }


__device__ int clamped_integer_max(float a, float b, float c, int low, int high)
{
    return min(max(float2int(ceil(max(max(a, b), c))), low), high);
}


__device__ int clamped_integer_min(float a, float b, float c, int low, int high) {
    return min(max(float2int(floor(min(min(a, b), c))), low), high);
}


__device__ void compute_edge_functions(const float px, const float py, const float m_inv[9], float values[3])
{
    for (int i = 0; i < 3; ++i)
    {
        const float a = m_inv[3 * i + 0];
        const float b = m_inv[3 * i + 1];
        const float c = m_inv[3 * i + 2];

        values[i] = a * px + b * py + c;
    }
}


__device__ void compute_unnormalized_matrix_inverse(
    const float a11, const float a12, const float a13,
    const float a21, const float a22, const float a23,
    const float a31, const float a32, const float a33, float m_inv[9])
{
    m_inv[0] = a22 * a33 - a32 * a23;
    m_inv[1] = a13 * a32 - a33 * a12;
    m_inv[2] = a12 * a23 - a22 * a13;
    m_inv[3] = a23 * a31 - a33 * a21;
    m_inv[4] = a11 * a33 - a31 * a13;
    m_inv[5] = a13 * a21 - a23 * a11;
    m_inv[6] = a21 * a32 - a31 * a22;
    m_inv[7] = a12 * a31 - a32 * a11;
    m_inv[8] = a11 * a22 - a21 * a12;

    // The first column of the unnormalized M^-1 contains intermediate values for det(M).
    const float det = a11 * m_inv[0] + a12 * m_inv[3] + a13 * m_inv[6];

    // Transfer the sign of the determinant.
    if (det < 0.0f)
    {
        for (int i = 0; i < 9; ++i)
        {
            m_inv[i] = -m_inv[i];
        }
    }
}


__device__ __forceinline__ bool pixel_is_inside_triangle(const float edge_values[3])
{
    // Check that the edge values are all non-negative and that at least one is positive (triangle is non-degenerate).
    // loose the constraint from 0.0 to -0.00001 to solve salt-pepper rendering
    float eps = -0.00001;
    return (edge_values[0] >= eps && edge_values[1] >= eps && edge_values[2] >= eps) && (edge_values[0] > 0 || edge_values[1] > 0 || edge_values[2] > 0);
}


__global__ void rasterize_triangles_cuda_forward_kernel(
    const torch::PackedTensorAccessor<float, 2, torch::RestrictPtrTraits, size_t> vertices,
    const torch::PackedTensorAccessor<int32_t, 2, torch::RestrictPtrTraits, size_t> triangles,
    const int image_width,
    const int image_height,
    torch::PackedTensorAccessor<int32_t, 2, torch::RestrictPtrTraits, size_t> px_triangle_ids,
    torch::PackedTensorAccessor<float, 3, torch::RestrictPtrTraits, size_t> px_barycentric_coordinates,
    torch::PackedTensorAccessor<float, 2, torch::RestrictPtrTraits, size_t> z_buffer,
    const int num_triangles
)
{
    const int triangle_id = threadIdx.x + blockIdx.x * blockDim.x;

    if (triangle_id >= num_triangles)
    {
        return;
    }

    const float half_image_width = 0.5 * image_width;
    const float half_image_height = 0.5 * image_height;

    float unnormalized_matrix_inverse[9];
    float b_over_w[3];

    const int v0_id = triangles[triangle_id][0];
    const int v1_id = triangles[triangle_id][1];
    const int v2_id = triangles[triangle_id][2];

    const float v0w = vertices[v0_id][3];
    const float v1w = vertices[v1_id][3];
    const float v2w = vertices[v2_id][3];

    // Early exit: if all w < 0, triangle is entirely behind the eye.
    if (v0w < 0 && v1w < 0 && v2w < 0)
    {
        return;
    }

    const float v0x = vertices[v0_id][0];
    const float v0y = vertices[v0_id][1];
    const float v1x = vertices[v1_id][0];
    const float v1y = vertices[v1_id][1];
    const float v2x = vertices[v2_id][0];
    const float v2y = vertices[v2_id][1];

    compute_unnormalized_matrix_inverse(v0x, v1x, v2x,
                                        v0y, v1y, v2y,
                                        v0w, v1w, v2w,
                                        unnormalized_matrix_inverse);

    // Initialize the bounding box to the entire screen.
    int left = 0, right = image_width, bottom = 0, top = image_height;
    // If the triangle is entirely inside the screen, project the vertices to
    // pixel coordinates and find the triangle bounding box enlarged to the
    // nearest integer and clamped to the image boundaries.
    if (v0w > 0 && v1w > 0 && v2w > 0)
    {
        const float p0x = (v0x / v0w + 1.0) * half_image_width;
        const float p1x = (v1x / v1w + 1.0) * half_image_width;
        const float p2x = (v2x / v2w + 1.0) * half_image_width;
        const float p0y = (v0y / v0w + 1.0) * half_image_height;
        const float p1y = (v1y / v1w + 1.0) * half_image_height;
        const float p2y = (v2y / v2w + 1.0) * half_image_height;
        left = clamped_integer_min(p0x, p1x, p2x, 0, image_width);
        right = clamped_integer_max(p0x, p1x, p2x, 0, image_width);
        bottom = clamped_integer_min(p0y, p1y, p2y, 0, image_height);
        top = clamped_integer_max(p0y, p1y, p2y, 0, image_height);
    }

    // Iterate over each pixel in the bounding box.
    for (int iy = bottom; iy < top; ++iy)
    {
        for (int ix = left; ix < right; ++ix)
        {
            const float px = ((ix + 0.5) / half_image_width) - 1.0;
            const float py = ((iy + 0.5) / half_image_height) - 1.0;

            compute_edge_functions(px, py, unnormalized_matrix_inverse, b_over_w);
            if (!pixel_is_inside_triangle(b_over_w))
            {
                continue;
            }

            const float one_over_w = b_over_w[0] + b_over_w[1] + b_over_w[2];
            const float b0 = b_over_w[0] / one_over_w;
            const float b1 = b_over_w[1] / one_over_w;
            const float b2 = b_over_w[2] / one_over_w;

            const float v0z = vertices[v0_id][2];
            const float v1z = vertices[v1_id][2];
            const float v2z = vertices[v2_id][2];
            // Since we computed an unnormalized w above, we need to recompute
            // a properly scaled clip-space w value and then divide clip-space z
            // by that.
            const float clip_z = b0 * v0z + b1 * v1z + b2 * v2z;
            const float clip_w = b0 * v0w + b1 * v1w + b2 * v2w;
            const float z = clip_z / clip_w;

            // Skip the pixel if it is farther than the current z-buffer pixel or beyond the near or far clipping plane.
            if (z < -1.0 || z > 1.0 || z > z_buffer[iy][ix])
            {
                continue;
            }

            atomicMin(&z_buffer[iy][ix], z);

            if (z == z_buffer[iy][ix])
            {
                px_triangle_ids[iy][ix] = triangle_id;
                px_barycentric_coordinates[iy][ix][0] = b0;
                px_barycentric_coordinates[iy][ix][1] = b1;
                px_barycentric_coordinates[iy][ix][2] = b2;
            }
        }
    }
}


std::vector<torch::Tensor> rasterize_triangles_cuda_forward(
    const torch::Tensor &vertices,
    const torch::Tensor &triangles,
    const int image_width,
    const int image_height,
    torch::Tensor &px_triangle_ids,  // image height * image width    int32  zeros
    torch::Tensor &px_barycentric_coordinates,  // image height * image width * 3    float32 ones require grad
    torch::Tensor &z_buffer  // image height * image width *     float32 ones
)
{
    const int num_triangles = triangles.size(0);
    const int threads = 512;
    const dim3 blocks = ((num_triangles - 1) / threads + 1);

    rasterize_triangles_cuda_forward_kernel<<<blocks, threads>>>(
        vertices.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>(),
        triangles.packed_accessor<int32_t, 2, torch::RestrictPtrTraits, size_t>(),
        image_width,
        image_height,
        px_triangle_ids.packed_accessor<int32_t, 2, torch::RestrictPtrTraits, size_t>(),
        px_barycentric_coordinates.packed_accessor<float, 3, torch::RestrictPtrTraits, size_t>(),
        z_buffer.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>(),
        num_triangles
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Error in rasterize_triangles_cuda_forward: %s\n", cudaGetErrorString(err));

    return {px_triangle_ids, px_barycentric_coordinates, z_buffer};
}


__global__ void rasterize_triangles_cuda_backward_kernel(
    const torch::PackedTensorAccessor<float, 2, torch::RestrictPtrTraits, size_t> vertices,
    const torch::PackedTensorAccessor<int32_t, 2, torch::RestrictPtrTraits, size_t> triangles,
    const torch::PackedTensorAccessor<int32_t, 2, torch::RestrictPtrTraits, size_t> triangle_ids,
    const torch::PackedTensorAccessor<float, 3, torch::RestrictPtrTraits, size_t> barycentric_coordinates,
    const torch::PackedTensorAccessor<float, 3, torch::RestrictPtrTraits, size_t> df_dbarycentric_coordinates,
    const int image_width,
    const int image_height,
    torch::PackedTensorAccessor<float, 2, torch::RestrictPtrTraits, size_t> df_dvertices
)
{
    const int pixel_id =  blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel_id >= image_width * image_height)
    {
        return ;
    }
    // We first loop over each pixel in the output image, and compute
    // dbarycentric_coordinate[0,1,2]/dvertex[0x, 0y, 1x, 1y, 2x, 2y].
    // Next we compute each value above's contribution to
    // df/dvertices, building up that matrix as the output of this iteration.

    const int ix = pixel_id % image_width;
    const int iy = pixel_id / image_width;

    // b0, b1, and b2 are the three barycentric coordinate values
    // rendered at pixel pixel_id.
    const float b0 = barycentric_coordinates[iy][ix][0];
    const float b1 = barycentric_coordinates[iy][ix][1];
    const float b2 = barycentric_coordinates[iy][ix][2];

    if (b0 + b1 + b2 < kDegenerateBarycentricCoordinatesCutoff())
    {
        return;
    }

    const float df_db0 = df_dbarycentric_coordinates[iy][ix][0];
    const float df_db1 = df_dbarycentric_coordinates[iy][ix][1];
    const float df_db2 = df_dbarycentric_coordinates[iy][ix][2];

    const int triangle_at_current_pixel = triangle_ids[iy][ix];

    // Extract vertex indices for the current triangle.
    const int v0_id = triangles[triangle_at_current_pixel][0];
    const int v1_id = triangles[triangle_at_current_pixel][1];
    const int v2_id = triangles[triangle_at_current_pixel][2];

    // Extract x,y,w components of the vertices' clip space coordinates.
    const float x0 = vertices[v0_id][0];
    const float y0 = vertices[v0_id][1];
    const float w0 = vertices[v0_id][3];

    const float x1 = vertices[v1_id][0];
    const float y1 = vertices[v1_id][1];
    const float w1 = vertices[v1_id][3];

    const float x2 = vertices[v2_id][0];
    const float y2 = vertices[v2_id][1];
    const float w2 = vertices[v2_id][3];

    // Compute pixel's NDC-s.

    const float px = 2 * (ix + 0.5f) / image_width - 1.0f;
    const float py = 2 * (iy + 0.5f) / image_height - 1.0f;

    // Baricentric gradients wrt each vertex coordinate share a common factor.
    const float db0_dx = py * (w1 - w2) - (y1 - y2);
    const float db1_dx = py * (w2 - w0) - (y2 - y0);
    const float db2_dx = -(db0_dx + db1_dx);
    const float db0_dy = (x1 - x2) - px * (w1 - w2);
    const float db1_dy = (x2 - x0) - px * (w2 - w0);
    const float db2_dy = -(db0_dy + db1_dy);
    const float db0_dw = px * (y1 - y2) - py * (x1 - x2);
    const float db1_dw = px * (y2 - y0) - py * (x2 - x0);
    const float db2_dw = -(db0_dw + db1_dw);

    // Combine them with chain rule.
    const float df_dx = df_db0 * db0_dx + df_db1 * db1_dx + df_db2 * db2_dx;
    const float df_dy = df_db0 * db0_dy + df_db1 * db1_dy + df_db2 * db2_dy;
    const float df_dw = df_db0 * db0_dw + df_db1 * db1_dw + df_db2 * db2_dw;

    // Values of edge equations and inverse w at the current pixel.
    const float edge0_over_w = x2 * db0_dx + y2 * db0_dy + w2 * db0_dw;
    const float edge1_over_w = x2 * db1_dx + y2 * db1_dy + w2 * db1_dw;
    const float edge2_over_w = x1 * db2_dx + y1 * db2_dy + w1 * db2_dw;
    const float w_inv = edge0_over_w + edge1_over_w + edge2_over_w;

    // All gradients share a common denominator.
    const float w_sqr = 1 / (w_inv * w_inv);

    // Gradients wrt each vertex share a common factor.
    const float edge0 = w_sqr * edge0_over_w;
    const float edge1 = w_sqr * edge1_over_w;
    const float edge2 = w_sqr * edge2_over_w;

    atomicAdd(&df_dvertices[v0_id][0], edge0 * df_dx);
    atomicAdd(&df_dvertices[v0_id][1], edge0 * df_dy);
    atomicAdd(&df_dvertices[v0_id][3], edge0 * df_dw);
    atomicAdd(&df_dvertices[v1_id][0], edge1 * df_dx);
    atomicAdd(&df_dvertices[v1_id][1], edge1 * df_dy);
    atomicAdd(&df_dvertices[v1_id][3], edge1 * df_dw);
    atomicAdd(&df_dvertices[v2_id][0], edge2 * df_dx);
    atomicAdd(&df_dvertices[v2_id][1], edge2 * df_dy);
    atomicAdd(&df_dvertices[v2_id][3], edge2 * df_dw);
}


torch::Tensor rasterize_triangles_cuda_backward(
    const torch::Tensor &vertices,
    const torch::Tensor &triangles,
    const torch::Tensor &triangle_ids,
    const torch::Tensor &barycentric_coordinates,
    const torch::Tensor &df_dbarycentric_coordinates,
    const int image_width,
    const int image_height,
    torch::Tensor &df_dvertices // num_vertex * 4 float32 zeros
)
{
    const int threads = 512;
    const dim3 blocks = ((image_width * image_height - 1) / threads + 1);

    rasterize_triangles_cuda_backward_kernel<<<blocks, threads>>>(
        vertices.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>(),
        triangles.packed_accessor<int32_t, 2, torch::RestrictPtrTraits, size_t>(),
        triangle_ids.packed_accessor<int32_t, 2, torch::RestrictPtrTraits, size_t>(),
        barycentric_coordinates.packed_accessor<float, 3, torch::RestrictPtrTraits, size_t>(),
        df_dbarycentric_coordinates.packed_accessor<float, 3, torch::RestrictPtrTraits, size_t>(),
        image_width,
        image_height,
        df_dvertices.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>()
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Error in rasterize_triangles_cuda_backward: %s\n", cudaGetErrorString(err));

    return df_dvertices;
}
