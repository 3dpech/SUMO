#include <opencv2/opencv.hpp>
#include<opencv2\core.hpp>
#include<opencv2\highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <numeric>

using namespace std;
using namespace cv;

IplImage* output_erode = 0;
IplImage* dst = 0;
IplImage* cvThreshold_XY = 0;

int main(int argc, char** argv)
{
	
	Mat vhimage = imread("Visual_Cargo_x1601_y7784_vh.TIF", CV_LOAD_IMAGE_GRAYSCALE); // vh - кроссполяризованное изображение; vv - параллельно поляризованное.
	Mat vvimage = imread("Visual_Cargo_x1601_y7784_vv.TIF", CV_LOAD_IMAGE_GRAYSCALE);

	int VH_sum = 0;
	int VH_sumpix = 0;
	int VH_z = 0; // область корабля
	
	//расчет среднего арифметического значения пикселей фона
	for (int i = 0; i < vhimage.rows; i = i + 2)// +2 для уменьшения шума фона
	{
		for (int j = 0; j < vhimage.cols; j = j + 2)
		{
			int A = (int)vhimage.at<uchar>(i, j);
			if ((i > (vhimage.rows / 3)) && (i < (vhimage.rows * 2 / 3)) && (j > (vhimage.cols / 3)) && (j < (vhimage.cols * 2 / 3)))// область корабля занимает 1/9 изображения
			{
				VH_z = VH_z + 4;
			}
			else
			{
				VH_sumpix = VH_sumpix + A;
				VH_sum++;
			}
		}
	}
	int VH_M = VH_sumpix / VH_sum;
	cout << "VH_M: " << VH_M << endl;

	//расчет среднеквадратичного отклонения пикселей фона
	double A_SKO = 0;
	for (int i = 0; i < vhimage.rows; i= i + 2)
	{
		for (int j = 0; j < vhimage.cols; j = j + 2)
		{
			int A = (int)vhimage.at<uchar>(i, j);
			if ((i > (vhimage.rows / 3)) && (i < (vhimage.rows * 2 / 3)) && (j > (vhimage.cols / 3)) && (j < (vhimage.cols * 2 / 3)))
			{
				VH_z = VH_z + 4;
			}
			else
			{
				A_SKO = A_SKO + (pow((A - VH_M), 2));
			}
		}
	}
	double VH_SKO = int(sqrt(A_SKO / (VH_sum - 1)));
	cout << "VH_SKO: " << VH_SKO << endl;

	int VV_sum = 0;
	int VV_sumpix = 0;
	int VV_z = 0;

	for (int i = 0; i < vvimage.rows; i = i + 2)
	{
		for (int j = 0; j < vvimage.cols; j = j + 2)
		{
			int B = (int)vvimage.at<uchar>(i, j);
			if ((i > (vvimage.rows / 3)) && (i < (vvimage.rows * 2 / 3)) && (j > (vvimage.cols / 3)) && (j < (vvimage.cols * 2 / 3)))
			{
				VV_z = VV_z + 4;
			}
			else
			{
				VV_sumpix = VV_sumpix + B;
				VV_sum++;
			}
		}
	}
	int VV_M = VV_sumpix / VV_sum;
	cout << "VV_M: " << VV_M << endl;

	double B_SKO = 0;
	for (int i = 0; i < vvimage.rows; i = i + 2)
	{
		for (int j = 0; j < vvimage.cols; j = j + 2)
		{
			int B = (int)vvimage.at<uchar>(i, j);
			if ((i > (vvimage.rows / 3)) && (i < (vvimage.rows * 2 / 3)) && (j > (vvimage.cols / 3)) && (j < (vvimage.cols * 2 / 3)))
			{
				VV_z = VV_z + 4;;
			}
			else
			{
				B_SKO = B_SKO + (pow((B - VV_M), 2));
			}
		}
	}
	int VV_SKO = int(sqrt(B_SKO / (VV_sum - 1)));
	cout << "VV_SKO: " << VV_SKO << endl;

	//Обнаружение пикселей корабля и их 8 соседей на изображении vh
	int min_i = 10000;
    int max_i = 0;
    int min_j = 10000;
    int max_j = 0;

    for (int i = 1; i < vhimage.rows - 1; i++)
    {
        for (int j = 1; j < vhimage.cols - 1; j++)
        {
			int A = vhimage.at<uchar>(i, j);
            if (A >= 38)// 38 - порог обнаружения
            {
				if (vhimage.at<uchar>(i - 1, j) >= (VH_M + 3 * VH_SKO))// VH_M + 3 * VH_SKO - порог кластеризации
                {
					if (vhimage.at<uchar>(i + 1, j) >= (VH_M + 3 * VH_SKO))
                    {
						if (vhimage.at<uchar>(i, j - 1) >= (VH_M + 3 * VH_SKO))
                        {
							if (vhimage.at<uchar>(i, j + 1) >= (VH_M + 3 * VH_SKO))
                            {
								if (vhimage.at<uchar>(i + 1, j - 1) >= (VH_M + 3 * VH_SKO))
                                {
									if (vhimage.at<uchar>(i + 1, j + 1) >= (VH_M + 3 * VH_SKO))
                                    {
										if (vhimage.at<uchar>(i - 1, j + 1) >= (VH_M + 3 * VH_SKO))
                                        {
											if (vhimage.at<uchar>(i - 1, j - 1) >= (VH_M + 3 * VH_SKO))
											{
												if (vhimage.at<uchar>(i, j) >= (VH_M + 5 * VH_SKO))// VH_M + 5 * VH_SKO - сигнатура
                                                {
													if (min_i > i) min_i = i;
                                                    if (max_i < i) max_i = i;
                                                    if (min_j > j) min_j = j;
                                                    if (max_j < j) max_j = j;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
 
	Mat OutImgVH;
	vhimage.copyTo(OutImgVH);

	vector<int> vecvh;

	for (int i = min_i; i < max_i; i++)
	{
		for (int j = min_j; j < max_j; j++)
		{
			if (vhimage.at<uchar>(i, j) < (VH_M + 5 * VH_SKO))
			{
				OutImgVH.at<uchar>(i, j) = 0;
			}
			else
			{
				vecvh.push_back(vhimage.at<uchar>(i, j));
				OutImgVH.at<uchar>(i, j) = 255;
			}
		}
	}

	for (int i = 0; i < vhimage.rows; i++)
	{
		for (int j = 0; j < vhimage.cols; j++)
		{
			if (i > min_i && i < max_i && j > min_j && j < max_j)
			{
				if (vhimage.at<uchar>(i, j) >= (VH_M + 5 * VH_SKO))
				{
					OutImgVH.at<uchar>(i, j) = 255;
				}
			}
			else
			{
				OutImgVH.at<uchar>(i, j) = 0;
			}
		}
	}

	//Обнаружение пикселей корабля и их 8 соседей на изображении vv
	for (int i = 1; i < vvimage.rows - 1; i++)
	{
		for (int j = 1; j < vvimage.cols - 1; j++)
		{
			int B = vvimage.at<uchar>(i, j);
			if (B >= 30)
			{
				if (vvimage.at<uchar>(i - 1, j) >= (VV_M + 3 * VV_SKO))
				{
					if (vvimage.at<uchar>(i + 1, j) >= (VV_M + 3 * VV_SKO))
					{
						if (vvimage.at<uchar>(i, j - 1) >= (VV_M + 3 * VV_SKO))
						{
							if (vvimage.at<uchar>(i, j + 1) >= (VV_M + 3 * VV_SKO))
							{
								if (vvimage.at<uchar>(i + 1, j - 1) >= (VV_M + 3 * VV_SKO))
								{
									if (vvimage.at<uchar>(i + 1, j + 1) >= (VV_M + 3 * VV_SKO))
									{
										if (vvimage.at<uchar>(i - 1, j + 1) >= (VV_M + 3 * VV_SKO))
										{
											if (vvimage.at<uchar>(i - 1, j - 1) >= (VV_M + 3 * VV_SKO))
											{
												if (vvimage.at<uchar>(i, j) >= (VV_M + 5 * VV_SKO))
												{
													if (min_i > i) min_i = i;
													if (max_i < i) max_i = i;
													if (min_j > j) min_j = j;
													if (max_j < j) max_j = j;
												}
											}
										}
									}
								}
							}
						}
					}
				}
			}
		}
	}

	Mat OutImgVV;
	vvimage.copyTo(OutImgVV);

	vector<int> vecvv;
	
	for (int i = min_i; i < max_i; i++)
	{
		for (int j = min_j; j < max_j; j++)
		{
			if (vvimage.at<uchar>(i, j) < (VV_M + 5 * VV_SKO))
			{
				OutImgVV.at<uchar>(i, j) = 0;
			}
			else
			{
				vecvv.push_back(vvimage.at<uchar>(i, j));
				OutImgVV.at<uchar>(i, j) = 255;
			}
		}
	}

	for (int i = 0; i < vvimage.rows; i++)
	{
		for (int j = 0; j < vvimage.cols; j++)
		{
			if (i > min_i && i < max_i && j > min_j && j < max_j)
			{
				if (vvimage.at<uchar>(i, j) >= (VV_M + 5 * VV_SKO))
				{
					OutImgVV.at<uchar>(i, j) = 255;
				}
			}
			else
			{
				OutImgVV.at<uchar>(i, j) = 0;
			}
		}
	}

	// Для включение тех пикслей, что были обнаружены на одном изображении, но не были на другом. 
	for (int i = min_i; i < max_i; i++)
	{
		for (int j = min_j; j < max_j; j++)
		{
			if (vvimage.at<uchar>(i, j) < (VV_M + 5 * VV_SKO) && vhimage.at<uchar>(i, j) >= (VH_M + 5 * VH_SKO) || (vvimage.at<uchar>(i, j) >= (VV_M + 5 * VV_SKO) && vhimage.at<uchar>(i, j) < (VH_M + 5 * VH_SKO)))
			{
				vecvv.push_back(vvimage.at<uchar>(i, j));
				OutImgVV.at<uchar>(i, j) = 255;
				vecvh.push_back(vhimage.at<uchar>(i, j));
				OutImgVH.at<uchar>(i, j) = 255;
			}
		}
	}

	cvNamedWindow("CFAR-out-vh", CV_WINDOW_NORMAL);
	imshow("CFAR-out-vh", OutImgVH);
	
	cvNamedWindow("CFAR-out-vv", CV_WINDOW_NORMAL);
	imshow("CFAR-out-vv", OutImgVV);

	cvNamedWindow("CFAR-VH", CV_WINDOW_NORMAL);
	imshow("CFAR-VH", vhimage);

	cvNamedWindow("CFAR-VV", CV_WINDOW_NORMAL);
	imshow("CFAR-VV", vvimage);

	Mat input;
	OutImgVV.copyTo(input);
	Mat output;
	Mat output1;
	Mat output2;
	Mat output3;
	Mat output4;
	Mat output5;
	Mat output6;
	Mat output12;
	Mat output34;
	Mat output56;
	Mat output1234;
	Mat output_0;
	Mat output_15;
	Mat output_30;
	Mat output_45;
	Mat output_60;
	Mat output_75;
	Mat output_90;
	Mat output_105;
	Mat output_120;
	Mat output_135;
	Mat output_150;
	Mat output_165;

	// создание структурирующего элемента для операции эрозии с шагом в 15 градусов
	unsigned char Mat0[25] =
	{
		0, 0, 1, 0, 0,
		0, 0, 1, 0, 0,
		0, 0, 1, 0, 0,
		0, 0, 1, 0, 0,
		0, 0, 1, 0, 0
	};
	Mat element_0 = Mat(Size(5, 5), CV_8UC1, Mat0);

	unsigned char Mat15[25] =
	{
		0, 0, 0, 1, 0,
		0, 0, 1, 0, 0,
		0, 0, 1, 0, 0,
		0, 0, 1, 0, 0,
		0, 1, 0, 0, 0
	};
	Mat element_15 = Mat(Size(5, 5), CV_8UC1, Mat15);

	unsigned char Mat30[25] =
	{
		0, 0, 0, 1, 0,
		0, 0, 0, 1, 0,
		0, 0, 1, 0, 0,
		0, 1, 0, 0, 0,
		0, 1, 0, 0, 0
	};
	Mat element_30 = Mat(Size(5, 5), CV_8UC1, Mat30);

	unsigned char Mat45[25] =
	{
		0, 0, 0, 0, 1,
		0, 0, 0, 1, 0,
		0, 0, 1, 0, 0,
		0, 1, 0, 0, 0,
		1, 0, 0, 0, 0
	};
	Mat element_45 = Mat(Size(5, 5), CV_8UC1, Mat45);

	unsigned char Mat60[25] =
	{
		0, 0, 0, 0, 0,
		0, 0, 0, 1, 1,
		0, 0, 1, 0, 0,
		1, 1, 0, 0, 0,
		0, 0, 0, 0, 0
	};
	Mat element_60 = Mat(Size(25, 25), CV_8UC1, Mat60);

	unsigned char Mat75[25] =
	{
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 1,
		0, 1, 1, 1, 0,
		1, 0, 0, 0, 0,
		0, 0, 0, 0, 0
	};
	Mat element_75 = Mat(Size(5, 5), CV_8UC1, Mat75);

	unsigned char Mat90[25] =
	{
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,
		1, 1, 1, 1, 1,
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0
	};
	Mat element_90 = Mat(Size(5, 5), CV_8UC1, Mat90);

	unsigned char Mat105[25] =
	{
		0, 0, 0, 0, 0,
		1, 0, 0, 0, 0,
		0, 1, 1, 1, 0,
		0, 0, 0, 0, 1,
		0, 0, 0, 0, 0
	};
	Mat element_105 = Mat(Size(5, 5), CV_8UC1, Mat105);

	unsigned char Mat120[25] =
	{
		0, 0, 0, 0, 0,
		1, 1, 0, 0, 0,
		0, 0, 1, 0, 0,
		0, 0, 0, 1, 1,
		0, 0, 0, 0, 0
	};
	Mat element_120 = Mat(Size(5, 5), CV_8UC1, Mat120);

	unsigned char Mat135[25] =
	{
		1, 0, 0, 0, 0,
		0, 1, 0, 0, 0,
		0, 0, 1, 0, 0,
		0, 0, 0, 1, 0,
		0, 0, 0, 0, 1
	};
	Mat element_135 = Mat(Size(5, 5), CV_8UC1, Mat135);

	unsigned char Mat150[25] =
	{
		0, 1, 0, 0, 0,
		0, 1, 0, 0, 0,
		0, 0, 1, 0, 0,
		0, 0, 0, 1, 0,
		0, 0, 0, 1, 0
	};
	Mat element_150 = Mat(Size(5, 5), CV_8UC1, Mat150);

	unsigned char Mat165[25] =
	{
		0, 1, 0, 0, 0,
		0, 0, 1, 0, 0,
		0, 0, 1, 0, 0,
		0, 0, 1, 0, 0,
		0, 0, 0, 1, 0
	};
	Mat element_165 = Mat(Size(5, 5), CV_8UC1, Mat165);
	
	//эрозия
	erode(input, output_0, element_0);
	erode(input, output_15, element_15);
	erode(input, output_30, element_30);
	erode(input, output_45, element_45);
	erode(input, output_60, element_60);
	erode(input, output_75, element_75);
	erode(input, output_90, element_90);
	erode(input, output_105, element_105);
	erode(input, output_120, element_120);
	erode(input, output_135, element_135);
	erode(input, output_150, element_150);
	erode(input, output_165, element_165);
	
	double alpha = 0.5;
	double beta = 0.5;

	//сложение изображений для получения 1 конечного результата
	addWeighted(output_0, alpha, output_15, beta, 0.0, output1);
	addWeighted(output_30, alpha, output_45, beta, 0.0, output2);
	addWeighted(output_60, alpha, output_75, beta, 0.0, output3);
	addWeighted(output_90, alpha, output_105, beta, 0.0, output4);
	addWeighted(output_120, alpha, output_135, beta, 0.0, output5);
	addWeighted(output_150, alpha, output_165, beta, 0.0, output6);

	addWeighted(output1, alpha, output2, beta, 0.0, output12);
	addWeighted(output3, alpha, output4, beta, 0.0, output34);
	addWeighted(output5, alpha, output6, beta, 0.0, output56);

	addWeighted(output12, alpha, output34, beta, 0.0, output1234);
	addWeighted(output1234, alpha, output56, beta, 0.0, output);

	cvNamedWindow("erode", CV_WINDOW_NORMAL);
	imshow("erode", output);

	// создаем вектор для порога, чтобы отсечь "лишние" пиксели
	vector<int> threshold;
	for (int i = 0; i < output.rows; i++)
	{
		for (int j = 0; j < output.cols; j++)
		{
			output.at<uchar>(i, j);
			if (output.at<uchar>(i, j) > 0)
			{
				threshold.push_back(output.at<uchar>(i, j));
			}
		}
	}

	// Находим среднее арифметическое и медианное значения пикселей корабля для создания порога - сохранения пикселей более высоких значений.
	int M = accumulate(threshold.begin(), threshold.end(), 0.0) / threshold.size();
	int B = threshold.front();
	int C = threshold.back();
	int Median = abs((B - C) / 2);

	//Заносим в порог большее значение из среднего и медианного 
	int K;
	if (Median <= M)
	{
		K = M;
	}
	else
	{
		K = Median;
	}

	output_erode = cvCloneImage(&(IplImage)output);

	dst = cvCreateImage(cvSize(output_erode->width, output_erode->height), IPL_DEPTH_8U, 1);

	cvThreshold(output_erode, dst, K, 255, CV_THRESH_BINARY);

	cvNamedWindow("cvThreshold", CV_WINDOW_NORMAL);
	cvShowImage("cvThreshold", dst);

	//определяем ориентацию корабля на изображении
	Mat ang = cvarrToMat(dst);
	vector<Point> pixels;
	Mat_<uchar>::iterator t = ang.begin<uchar>();
	Mat_<uchar>::iterator e = ang.end<uchar>();
	for (; t != e; ++t)
		if (*t)
			pixels.push_back(t.pos());
	RotatedRect box_XY = minAreaRect(Mat(pixels));
	double angle = box_XY.angle;
	if (angle < 45.)
		angle += 90.;
	cout << "angle: " << angle << endl;
	
	CvMat* rot_mat = cvCreateMat(2, 3, CV_32FC1);
	CvMat* warp_mat = cvCreateMat(2, 3, CV_32FC1);
	IplImage* rotate = 0;
	rotate = cvCloneImage(dst);
	CvPoint2D32f center = cvPoint2D32f(dst->width / 2, dst->height / 2);
	double scale = 1;
	cv2DRotationMatrix(center, angle, scale, rot_mat);
	// выполняем вращение
	cvWarpAffine(dst, rotate, rot_mat);
	/*cvNamedWindow("cvWarpAffine", WINDOW_NORMAL);
	cvShowImage("cvWarpAffine", rotate);*/

	Mat m = cvarrToMat(rotate);
	Mat dst_Y;
	Mat dst_X;
	Mat dst_XY;
	
	//операция opening по главным осям X и Y
	morphologyEx(m, dst_Y, MORPH_OPEN, element_0);
	morphologyEx(m, dst_X, MORPH_OPEN, element_90);

	//складываем результаты
	addWeighted(dst_Y, alpha, dst_X, beta, 0.0, dst_XY);

	cvNamedWindow("CV_MOP_OPEN", CV_WINDOW_NORMAL);
	imshow("CV_MOP_OPEN", dst_XY);
	
	//создаем новый порог и отсекаем пиксели по меньшему значению из среднего и медианного
	vector<int> threshold_XY;
	for (int i = 0; i < dst_XY.rows; i++)
	{
		for (int j = 0; j < dst_XY.cols; j++)
		{
			dst_XY.at<uchar>(i, j);
			if (dst_XY.at<uchar>(i, j) > 0)
			{
				threshold_XY.push_back(dst_XY.at<uchar>(i, j));
			}
		}
	}

	int M_XY = accumulate(threshold_XY.begin(), threshold_XY.end(), 0.0) / threshold_XY.size();
	int B_XY = threshold_XY.front();
	int C_XY = threshold_XY.back();
	int Median_XY = abs((B_XY - C_XY) / 2);

	int K_XY;
	if (Median_XY <= M_XY)
	{
		K_XY = Median_XY;
	}
	else
	{
		K_XY = M_XY;
	}
	
	IplImage* dst_threshold = cvCloneImage(&(IplImage)dst_XY);
	
	cvThreshold_XY = cvCreateImage(cvSize(dst_threshold->width, dst_threshold->height), IPL_DEPTH_8U, 1);

	cvThreshold(dst_threshold, cvThreshold_XY, K_XY, 255, CV_THRESH_BINARY);

	cvNamedWindow("cvThreshold_XY", CV_WINDOW_NORMAL);
	cvShowImage("cvThreshold_XY", cvThreshold_XY);

	//создаем рамку для получившегося объекта, которая будет определять его размер
	// 1 пиксель = 10 метров
	Mat result = cvarrToMat(cvThreshold_XY);

	vector<Point> points;
	Mat_<uchar>::iterator it = result.begin<uchar>();
	Mat_<uchar>::iterator end = result.end<uchar>();
	for (; it != end; ++it)
		if (*it)
			points.push_back(it.pos());
	RotatedRect box = minAreaRect(Mat(points));
	Point2f vertices[4];
	box.points(vertices);
	for (int i = 0; i < 4; ++i)
	{
		line(result, vertices[(i)], vertices[(i + 1) % 4], Scalar(150, 0, 0), 1, 4);
	}
	//выводим координаты рамки
	cout << "x1_y1: " << vertices[(0)] << endl;
	cout << "x1_y2: " << vertices[(1)] << endl;
	cout << "x2_y2: " << vertices[(2)] << endl;
	cout << "x2_y1: " << vertices[(3)] << endl;

	cvNamedWindow("Result", CV_WINDOW_NORMAL);
	imshow("Result", result);

	cvWaitKey();
	cvDestroyAllWindows;

	return 0;
}