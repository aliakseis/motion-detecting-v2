// motion-detecting.cpp : This file contains the 'main' function. Program execution begins and ends there.
//


#include "pch.h"

#include "nanoflann.hpp"

#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>

#include <future>

using namespace cv;
using namespace std;

using namespace nanoflann;

//////////////////////////////////////////////////////////////////////////////

// https://github.com/jmtilli/fastdiv

struct fastdivctx {
    uint32_t mult;
    uint32_t mod;
    //uint8_t shift1 : 1;
    //uint8_t shift2 : 7;
    uint32_t shift1, shift2;
};

constexpr inline
uint8_t ilog(uint32_t i)
{
    uint8_t result = 0;
    while (i >>= 1)
    {
        result++;
    }
    return result;
}

//static 
inline void init_fastdivctx(fastdivctx *ctx, uint32_t divisor)
{
    uint8_t ilogd = ilog(divisor);
    int power_of_2 = (divisor & (divisor - 1)) == 0;
    if (divisor == 0 || divisor >= (1U << 31))
    {
        abort(); // Not supported
    }
    if (power_of_2)
    {
        ctx->shift1 = 0;
    }
    else
    {
        ctx->shift1 = 1;
    }
    ctx->shift2 = ilogd;
    ctx->mod = divisor;
    ctx->mult = (1ULL << (32 + ctx->shift1 + ctx->shift2)) / divisor + 1;
}

//static 
inline uint32_t fastmod(const fastdivctx *ctx, uint32_t eax)
{
    uint64_t edxeax = ((uint64_t)eax) * ctx->mult;
    uint32_t edx = edxeax >> 32;
    uint32_t eaxorig = eax;
    eax -= edx;
    eax >>= (ctx->shift1);
    eax += edx;
    eax >>= (ctx->shift2);
    edx = ctx->mod*eax;
    return eaxorig - edx;
}

//static 
inline uint32_t fastdiv(const fastdivctx *ctx, uint32_t eax)
{
    uint64_t edxeax = ((uint64_t)eax) * ctx->mult;
    uint32_t edx = edxeax >> 32;
    eax -= edx;
    eax >>= (ctx->shift1);
    eax += edx;
    eax >>= (ctx->shift2);
    return eax;
}

//static 
inline void fastdivmod(const fastdivctx *ctx, uint32_t eax,
    uint32_t *div, uint32_t *mod)
{
    uint64_t edxeax = ((uint64_t)eax) * ctx->mult;
    uint32_t edx = edxeax >> 32;
    uint32_t eaxorig = eax;
    eax -= edx;
    eax >>= (ctx->shift1);
    eax += edx;
    eax >>= (ctx->shift2);
    *div = eax;
    edx = ctx->mod*eax;
    *mod = eaxorig - edx;
}

template<int divisor>
inline void fastdivmod(uint32_t eax,
    uint32_t *div, uint32_t *mod)
{
    enum { shift1 = (divisor & (divisor - 1)) != 0 };
    enum { shift2 = ilog(divisor) };
    constexpr uint32_t mult = (1ULL << (32 + shift1 + shift2)) / divisor + 1;

    uint64_t edxeax = ((uint64_t)eax) * mult;
    uint32_t edx = edxeax >> 32;
    uint32_t eaxorig = eax;
    eax -= edx;
    eax >>= (shift1);
    eax += edx;
    eax >>= (shift2);
    *div = eax;
    edx = divisor * eax;
    *mod = eaxorig - edx;
}


//////////////////////////////////////////////////////////////////////////////

enum { DIMENSION = 5 };

enum { ADDITIONAL = 2 };

enum { NUM_ATTRIBUTES = DIMENSION * DIMENSION + ADDITIONAL };

typedef float AttributeType;

class PointsProvider
{
public:
    PointsProvider(const cv::Mat* mat)
        : m_mat(mat)
        , m_numRows(mat->rows - DIMENSION + 1)
        , m_numCols(mat->cols - DIMENSION + 1)
        , m_coeffs(m_numRows * m_numCols, 1.f)
    {
        init_fastdivctx(&m_fastdivctx, m_numCols);
    }

    size_t kdtree_get_point_count() const
    {
        return m_numRows * m_numCols;
    }

    // Returns the dim'th component of the idx'th point in the class:
    // Since this is inlined and the "dim" argument is typically an immediate value, the
    //  "if/else's" are actually solved at compile time.
    float kdtree_get_pt(const size_t idx, const size_t dim) const
    {
        //const auto x = idx % m_numCols;
        //const auto y = idx / m_numCols;

        uint32_t x, y;
        fastdivmod(&m_fastdivctx, idx, &y, &x);
        
        switch (dim)
        {
        case 0: return double(y + DIMENSION / 2) / m_mat->rows;
        case 1: return double(x + DIMENSION / 2) / m_mat->cols;
        }

        const double coeff = get_coeff(idx);
        
        const double v = do_kdtree_get_pt(x, y, dim);

        return v / coeff;
    }

    // Optional bounding-box computation: return false to default to a standard bbox computation loop.
    //   Return true if the BBOX was already computed by the class and returned in "bb" so it can be avoided to redo it again.
    //   Look at bb.size() to find out the expected dimensionality (e.g. 2 or 3 for point clouds)
    template <class BBOX>
    bool kdtree_get_bbox(BBOX& /* bb */) const { return false; }

private:
    float get_coeff(const size_t idx) const
    {
        float result = m_coeffs[idx];
        if (result == 1.f)
        {
            double sq_sum = 0;
            for (int dim = 0; dim < NUM_ATTRIBUTES - ADDITIONAL; ++dim)
            {
                const auto v = do_kdtree_get_pt(idx, dim + ADDITIONAL);
                sq_sum += v * v;
            }

            if (sq_sum > 0)
            {
                result = sqrt(sq_sum);
                const_cast<float&>(m_coeffs[idx]) = result;
            }
        }
        return result;
    }
    float do_kdtree_get_pt(const size_t idx, const size_t dim) const
    {
        //const auto x = idx % m_numCols;
        //const auto y = idx / m_numCols;

        uint32_t x, y;
        fastdivmod(&m_fastdivctx, idx, &y, &x);
        return do_kdtree_get_pt(x, y, dim);
    }
    //float do_kdtree_get_pt(uint32_t x, uint32_t y, const size_t dim) const
    //{
    //    const double v = m_mat->at<uchar>(
    //        y + ((dim - ADDITIONAL) / DIMENSION),
    //        x + ((dim - ADDITIONAL) % DIMENSION));
    //    return v;
    //}
    float do_kdtree_get_pt(uint32_t x, uint32_t y, size_t dim) const
    {
        uint32_t div, mod;
        fastdivmod<DIMENSION>(dim - ADDITIONAL, &div, &mod);

        const double v = m_mat->at<uchar>(y + div, x + mod);
        return v;
    }

private:
    const cv::Mat* m_mat;
    int m_numRows;
    int m_numCols;
    std::vector<float> m_coeffs;
    fastdivctx m_fastdivctx;
};


    // construct a kd-tree index:
typedef KDTreeSingleIndexAdaptor<
    L2_Simple_Adaptor<float, PointsProvider >,
    PointsProvider,
    NUM_ATTRIBUTES /* dim */
> my_kd_tree_t;


//////////////////////////////////////////////////////////////////////////////


// Function to compute the optical flow map
void drawOpticalFlow(const std::vector<std::pair<Point, Point>>& shifts, Mat& flowImageGray)
{
    int stepSize = 16;
    Scalar color = Scalar(0, 255, 0);

    for (const auto& pair : shifts)
    {
        // Circles to indicate the uniform grid of points
        int radius = 2;
        int thickness = -1;
        circle(flowImageGray, pair.first, radius, color, thickness);
        line(flowImageGray, pair.first, pair.second, color);
    }
}


int main(int argc, char** argv)
{
    try
    {
        // set default values for tracking algorithm and video
    string videoPath = (argc == 2) ? argv[1] : "videos/run.mp4";


    // create a video capture object to read videos
    cv::VideoCapture cap(videoPath);

    if (!cap.isOpened())
    {
        cerr << "Unable to open the file. Exiting!" << endl;
        return -1;
    }

    char ch;
    Mat curGray, prevGray, flowImageGray, frame;
    string windowName = "Optical Flow";
    namedWindow(windowName, 1);
    //float scalingFactor = 0.75;

    // Iterate until the user presses the Esc key
    while (true)
    {
        // Capture the current frame
        cap >> frame;

        if (frame.empty())
            break;

        // Resize the frame
        //resize(frame, frame, Size(), scalingFactor, scalingFactor, INTER_AREA);

        // Convert to grayscale
        cvtColor(frame, curGray, COLOR_BGR2GRAY);

        // Check if the image is valid
        if (prevGray.data)
        {
            PointsProvider provider(&prevGray);

            my_kd_tree_t infos(NUM_ATTRIBUTES, provider);

            //infos.buildIndex();
            infos.fastBuildIndex();

            const auto numRows = prevGray.rows - DIMENSION + 1;
            const auto numCols = prevGray.cols - DIMENSION + 1;

            typedef std::vector<std::pair<Point, Point>> MapType;

            auto lam = [&infos, &curGray](int yBegin, int yEnd) {
                
                const auto numCols = curGray.cols - DIMENSION + 1;

                MapType shifts;

                // searching
                //for (int y = 0; y < numRows; ++y)
                for (int y = yBegin; y < yEnd; ++y)
                {
                    for (int x = 0; x < numCols; ++x)
                    {
                        if ((y & 7) || (x & 7))
                            continue;

                        AttributeType pos[NUM_ATTRIBUTES];
                        unsigned int sq_sum = 0;
                        for (int i = 0; i < DIMENSION; ++i)
                            for (int j = 0; j < DIMENSION; ++j)
                            {
                                const auto v = curGray.at<uchar>(y + i, x + j);
                                pos[i * DIMENSION + j + ADDITIONAL] = v;
                                sq_sum += v * v;
                            }

                        pos[0] = 0;
                        pos[1] = 0;

                        if (sq_sum > 0)
                        {
                            const auto coeff = sqrt(sq_sum);
                            for (auto& v : pos)
                                v /= coeff;
                        }

                        pos[0] = float(y + DIMENSION / 2) / curGray.rows;
                        pos[1] = float(x + DIMENSION / 2) / curGray.cols;


                        size_t num_results = 2;
                        std::vector<size_t>   ret_index(num_results);
                        std::vector<float> out_dist_sqr(num_results);

                        num_results = infos.knnSearch(&pos[0], num_results, &ret_index[0], &out_dist_sqr[0]);

                        // In case of less points in the tree than requested:
                        ret_index.resize(num_results);
                        out_dist_sqr.resize(num_results);


                        if (out_dist_sqr[1] > 0 && out_dist_sqr[1] > out_dist_sqr[0] * 1.2)
                        {
                            Point ptFrom((ret_index[0] % numCols) + DIMENSION / 2, (ret_index[0] / numCols) + DIMENSION / 2);
                            Point ptTo(x + DIMENSION / 2, y + DIMENSION / 2);
                            if (std::abs(ptTo.x - ptFrom.x) > 1 || std::abs(ptTo.y - ptFrom.y) > 1)
                            {
                                shifts.push_back({ ptFrom, ptTo });
                            }
                        }
                    }
                }

                return shifts;
            };

            enum { NUM_THREADS = 16 };

            std::vector<std::future<MapType>> proxies;

            for (int i = 0; i < NUM_THREADS; ++i)
            {
                proxies.push_back(std::async(std::launch::async, lam, 
                    (numRows * i) / NUM_THREADS,
                    (numRows * (i + 1)) / NUM_THREADS));
            }

            MapType shifts;

            for (auto& p : proxies)
            {
                auto v = p.get();
                shifts.insert(shifts.end(), std::make_move_iterator(v.begin()), std::make_move_iterator(v.end()));
            }

            // Convert to 3-channel RGB
            cvtColor(curGray, flowImageGray, COLOR_GRAY2BGR);

            // Draw the optical flow map
            drawOpticalFlow(shifts, flowImageGray);

            // Display the output image
            imshow(windowName, flowImageGray);
        }

        // Break out of the loop if the user presses the Esc key
        ch = waitKey(10);
        if (ch == 27)
            break;

        // Swap previous image with the current image
        std::swap(prevGray, curGray);
    }

    return 0;
    }
    catch (const std::exception& ex)
    {
        std::cerr << "Fatal: " << ex.what() << '\n';
        return 1;
    }
}
