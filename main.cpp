#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
//C
#include <stdio.h>
//C++
#include <iostream>
#include <sstream>
#include "tinyxml2.h"
//windows
#include <Windows.h>
#include <time.h>
#include "dhdvr.h"

using namespace cv;
using namespace std;
#pragma comment(lib, "winmm.lib")

#define NVR 0
#define IPC 1
#define FILE 3
#define CAM 2
int device;
// Global variables
Mat frame; //current frame
Mat frameshow;
//Mat fgMaskKNN; //fg mask fg mask generated by KNN method
Mat fgMaskMOG2; //fg mask fg mask generated by MOG2 method
//Mat bgKNN;
Mat bgMOG2; 
//Mat roi;//����Ȥ����
Rect roiRect;//����Ȥ����ľ���ֵ
Ptr<BackgroundSubtractor> pMOG2; //MOG2 Background subtractor
Ptr<BackgroundSubtractor> pKNN; //KNN Background subtractor
int keyboard; //input from keyboard
Mat temp;
VideoCapture cap;
//DHDVR* pdhdvr;
HANDLE hGetFrameEvent;
HANDLE hWarningEvent;
int fps;
int nframe;//�ڼ�֡
int	delay;//֡����

long whiteNum;
long sumNum;
double ratio;
stringstream ss;
const int niters = 3;
HANDLE hThread;//thread
HANDLE hThread2;
bool bgm_update_flag;//�Ƿ����bgm
Size imSize;
Size maskSize;
int scale;

vector<vector<cv::Point>> contours;//��ͨ����
vector<cv::Point> maxContour;//�����ͨ����
Rect maskbbox;
Rect bbox;

time_t startupTime;
time_t t1,t2;
const int invation_interval = 3000;//

double dx, dy;
double thresholdx, thresholdy;
int totalTime;
bool onsubtract_flag;
bool ontrack_flag;
bool oninvadedct_flag;

//�ǻ����
const int hover_interval = 10000;//����ʱ��
bool onhoverdct_flag;

int dctInterval;
int lastDctInterval;
int lastBgmUpdateTime;

vector<Point> points;

int drawbox_flag;
Rect box;
Point bd;
int pic_num = 1;

static void onMouse(int event, int x, int y, int flags, void *userdata)
{
	Rect *pbox = (Rect*)userdata;
	switch (event)
	{

	case EVENT_MOUSEMOVE:
		pbox->width = abs(bd.x - x);
		pbox->height = abs(bd.y - y);
		x > bd.x ? x : pbox->x = x;
		y > bd.y ? y : pbox->y = y;
		//x > pbox->x ? x : pbox->x = x;
		//y > pbox->y ? y : pbox->y = y;
		break;
		//cout << x << ','<< y << endl;
	case EVENT_LBUTTONDOWN:
		bd.x = x;
		bd.y = y;
		pbox->x = x;
		pbox->y = y;
		pbox->width = 0;
		pbox->height = 0;
		drawbox_flag = EVENT_LBUTTONDOWN;
		break;
	case EVENT_LBUTTONUP:
		//SetEvent(hBBoxReadyEvent);
		if (drawbox_flag == EVENT_LBUTTONDOWN)
		{
			drawbox_flag = EVENT_LBUTTONUP;
		}
		roiRect = *pbox;
		ontrack_flag = false;
		//points.clear();
		//;
		break;
	case EVENT_RBUTTONDOWN:
		//ResetEvent(hBBoxReadyEvent);
		pbox->x = -1;
		pbox->y = -1;
		pbox->width = 0;
		pbox->height = 0;
		drawbox_flag = false;
		pic_num = 1;
		roiRect.x=0;
		roiRect.y = 0;
		roiRect.width = imSize.width;
		roiRect.height = imSize.height;
		break;
	default:
		break;
	}

}

DWORD WINAPI WarningFun(LPVOID pM){
	while (1)
	{
		WaitForSingleObject(hWarningEvent, INFINITE);
		ResetEvent(hWarningEvent);
	}
	return 0;
}
DWORD WINAPI ThreadFun(LPVOID pM)
{
	while ((char)keyboard != 'q' && (char)keyboard != 27)
	{
		WaitForSingleObject(hGetFrameEvent,INFINITE);
		//printf("���̵߳��߳�ID��Ϊ��%d\n���߳����Hello World\n", GetCurrentThreadId());
		frame.copyTo(temp);
		resize(temp, temp, maskSize);
		thresholdx = imSize.width / (invation_interval / 1000)/2;
		thresholdy = imSize.height / (invation_interval / 1000)/2;
		//resize(temp, temp, Size(160, 120));
		//GaussianBlur(frame,frame,Size(3,3),0);
		//update the background model
		//medianBlur(frame, frame, 5);
		//pKNN->apply(temp, fgMaskKNN, bgm_update_flag ? -1 : 0);
		pMOG2->apply(temp, fgMaskMOG2, bgm_update_flag ? -1 : 0);
		//pKNN->getBackgroundImage(bgKNN);
		pMOG2->getBackgroundImage(bgMOG2);
		//dilate(fgMaskMOG2, fgMaskMOG2, Mat()); 
		erode(fgMaskMOG2, fgMaskMOG2, Mat(), Point(-1, -1), 1);
		dilate(fgMaskMOG2, fgMaskMOG2, Mat(), Point(-1, -1), 1);
		//erode(fgMaskKNN, fgMaskKNN, Mat(), Point(-1, -1), 1);
		//dilate(fgMaskKNN, fgMaskKNN, Mat(), Point(-1, -1), 1);
		whiteNum = countNonZero(fgMaskMOG2);
		//whiteNum = countNonZero(fgMaskKNN);
		sumNum = fgMaskMOG2.rows*fgMaskMOG2.cols;

		//sumNum = fgMaskKNN.rows*fgMaskKNN.cols;
		ratio = double(whiteNum) / double(sumNum);

		if (ratio>0.002&&onsubtract_flag)
		{
			//fgMaskKNN.copyTo(temp);
			fgMaskMOG2.copyTo(temp);

			findContours(temp, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
			// Ѱ�������ͨ��
			double maxArea = 0;
			double area = 0;
			for (int i = 0; i < contours.size(); i++)
			{
				double area = cv::contourArea(contours[i]);
				if (area > maxArea)
				{
					maxArea = area;
					maxContour = contours[i];
				}
			}
			maskbbox = cv::boundingRect(maxContour);
			//���ּ��
			if (oninvadedct_flag)
			{
				if (double(maskbbox.area()) / double(maskSize.area())>0.01)
				{
					bbox.x = maskbbox.x*scale;
					bbox.y = maskbbox.y*scale;
					bbox.width = maskbbox.width*scale;
					bbox.height = maskbbox.height*scale;
					Point p(bbox.x + bbox.width / 2, bbox.y + bbox.height / 2);
					if (roiRect.contains(p))
					{
						points.push_back(p);

						if (!ontrack_flag){
							ontrack_flag = true;
							points.clear();
							t1 = 0;
							t2 = 0;
							t1 = timeGetTime();

						}
						else
						{
							t2 = timeGetTime();
							int timeInterval = t2 - t1;
							if (timeInterval>500)
							{
								dx = (points.back().x - points.front().x) / double(timeInterval / 1000);
								dy = (points.back().y - points.front().y) / double(timeInterval / 1000);
								if (dx>thresholdx)
								{
									time_t t = time(0); //��������time(NULL)
									char tmp[64];
									strftime(tmp, sizeof(tmp), "%c", localtime(&t));
									cout << "���������" << "\t" << tmp << endl;
									putText(frameshow, "Invader Left", Point(10, 30), FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
									//SetEvent(hWarningEvent);
								}
								else if (dx<-thresholdx)
								{
									time_t t = time(0); //��������time(NULL)
									char tmp[64];
									strftime(tmp, sizeof(tmp), "%c", localtime(&t));
									cout << "���ұ�����" << "\t" << tmp << endl;
									putText(frameshow, "Invader Right", Point(frameshow.cols/2, 30), FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
									//SetEvent(hWarningEvent);
								}
							}
							if (t2 - t1>invation_interval)
							{
								ontrack_flag = false;
							}
						}
					} 
					
					rectangle(frameshow, bbox, Scalar(255, 0, 0), 2);

				}
			}
			//�ǻ����
			else if (onhoverdct_flag)
			{
				if (double(maskbbox.area()) / double(maskSize.area())>0.01)
				{
					if (!ontrack_flag)
					{
						t1 = timeGetTime();
						lastBgmUpdateTime = t1;
						ontrack_flag = true;
					}
					else
					{
						t2 = timeGetTime();
						lastDctInterval = dctInterval;
						dctInterval = t2 - t1;
						if (dctInterval - lastDctInterval>5000)//5����û��������֣����������¼�
						{
							ontrack_flag = false;
							t1 = 0;
							t2 = 0;
							lastDctInterval = 0;
							dctInterval = 0;
							lastBgmUpdateTime = 0;
							bgm_update_flag = 1;
							points.clear();
						}
						else if (dctInterval > hover_interval)//�������һ�γ��ֳ�����ֵ
						{
							bgm_update_flag = 0;
								time_t t = time(0); //������˵һ�º�������time(NULL),ע�⣡����
								char tmp[64];
								strftime(tmp, sizeof(tmp), "%c", localtime(&t));
								cout << "��������Ŀ�궺��\t" <<tmp << endl;
								putText(frameshow, "Hovering!!!", Point(10, 30), FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
								if (t2-lastBgmUpdateTime>hover_interval/10)
								{
									lastBgmUpdateTime = t2;
									bgm_update_flag = 1;
								}
								//SetEvent(hWarningEvent);
						}
						//dx = (points.back().x - points.front().x) / double(timeInterval / 1000);
						//dy = (points.back().y - points.front().y) / double(timeInterval / 1000);
						}
					}
					bbox.x = maskbbox.x*scale;
					bbox.y = maskbbox.y*scale;
					bbox.width = maskbbox.width*scale;
					bbox.height = maskbbox.height*scale;
					Point p(bbox.x + bbox.width / 2, bbox.y + bbox.height / 2);
					points.push_back(p);
					rectangle(frameshow, bbox, Scalar(0, 0, 255), 2);
					//if (t2 - t1>INTERVAL)
					//{
					//	ontrack_flag = false;
					//	points.clear();
					//	t1 = 0;
					//	t2 = 0;
					//}
			}
		}
			//cout << double(maskbbox.area()) / double(maskSize.area()) << endl;
		//putText(frameshow, ss.str(), cv::Point(10, 30), FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 0, 255), 2);
		if (!points.empty()&&oninvadedct_flag)
		{
			polylines(frameshow, points, false, Scalar(255, 0, 0), 2);
		}
		imshow("Frame", frameshow);
		//imshow("FG Mask KNN", fgMaskKNN);
		//imshow("FG Mask MOG 2", fgMaskMOG2);
		//imshow("BG KNN", bgKNN);
		//imshow("BG MOG 2", bgMOG2);
		ResetEvent(hGetFrameEvent);
		//keyboard = waitKey(delay);
	}
	return 0;
}


int main(int argc, char* argv[])
{
	//print help information
	//help();
	//check for the input parameter correctness
	//if (argc != 3) {
	//	cerr << "Incorret input list" << endl;
	//	cerr << "exiting..." << endl;
	//	return EXIT_FAILURE;
	//}
	//create GUI windows

	oninvadedct_flag = false;
	onhoverdct_flag = !oninvadedct_flag;

	DHDVR *pdhdvr = nullptr;
	//device = IPC;
	tinyxml2::XMLDocument doc;
	tinyxml2::XMLElement* root;
	if (doc.LoadFile("config.xml") == tinyxml2::XML_SUCCESS)
	{
		root = doc.FirstChildElement("device");
	}
	else
	{
		return -1;
	}
	const char* type = root->FirstChildElement("type")->GetText();
	string typeStr(type);

	if (typeStr=="IPC")
	{
		const char* rtsp = root->FirstChildElement("rtsp")->GetText();
		string rtspStr(rtsp);
		device = IPC;
		cout << rtspStr << endl;
		cap.open(rtspStr);
	}

	if (typeStr=="NVR")
	{
		const char* ip = root->FirstChildElement("ip")->GetText();
		const char* port = root->FirstChildElement("port")->GetText();
		const char* username = root->FirstChildElement("username")->GetText();
		const char* password;
		const char* channel = root->FirstChildElement("channel")->GetText();;
		if (root->FirstChildElement("password")->NoChildren())
		{
			password = "";
		}
		else
		{
			password = root->FirstChildElement("password")->GetText();
		}
		device = NVR;
		cout << "NVR" << endl;
		cout << ip<<":"<<port << endl;
		pdhdvr = new DHDVR(ip, stoi(port), username, password, stoi(channel));
	}
	//device = NVR;
	
	




	string path = "C:\\Users\\yulie\\Desktop\\video\\1.avi";

	if (device==FILE)
	{
		cap.open(path);
	}
	else if (device==CAM)
	{
		cap.open(0);
	}

	bgm_update_flag = true;
	onsubtract_flag = false;
	ontrack_flag = false;

	nframe = 0;
	hWarningEvent = CreateEvent(NULL, TRUE, FALSE, NULL);
	hGetFrameEvent =CreateEvent(NULL, TRUE, FALSE, NULL);
	frame.create(Size(320,240),CV_8UC3);
	//fgMaskKNN.create(Size(320, 240), CV_8UC3);
	fgMaskMOG2.create(Size(320, 240), CV_8UC3);
	namedWindow("Frame",CV_WINDOW_KEEPRATIO);
	//namedWindow("FG Mask KNN",CV_WINDOW_KEEPRATIO);
	//namedWindow("FG Mask MOG 2", CV_WINDOW_KEEPRATIO);
	//namedWindow("BG KNN", CV_WINDOW_KEEPRATIO);
	//namedWindow("BG MOG 2", CV_WINDOW_KEEPRATIO);
	//create Background Subtractor objects
	//pKNN = createBackgroundSubtractorKNN();
	setMouseCallback("Frame", onMouse, (void*)&box);

	pMOG2 = createBackgroundSubtractorMOG2(); //MOG2 approach
	//string path = "C:\\Users\\yulie\\Desktop\\video\\1.avi";
	//cap.open("rtsp://192.168.1.101:554/onvif/live/1");
	//cap.open(path);

	//if (!cap.isOpened()){
	//	//error in opening the video input
	//	cerr << "Unable to open video file: " << path << endl;
	//	exit(EXIT_FAILURE);
	//}


	//CreateThread();

	while ((char)keyboard != 'q' && (char)keyboard != 27)
	{
		//cap >> frame;
		if (device == NVR)
		{
			*pdhdvr >> frame;
		}
		else
		{
			cap >> frame;
		}
		
		if (frame.empty())
		{
			cerr << "Unable to read next frame." << endl;
			cerr << "Exiting..." << endl;

			exit(EXIT_FAILURE);
		}
		frame.copyTo(frameshow);
		nframe++;
		if (nframe == 1)
		{
			imSize = frame.size();
			//
			roiRect.x = 0;
			roiRect.y = 0;
			roiRect.width = imSize.width;
			roiRect.height = imSize.height;
			scale = imSize.width / 160;
			scale = 1;
			maskSize.width = imSize.width / scale;
			maskSize.height = imSize.height / scale;
			//fps = cap.get(CAP_PROP_FPS);
			//delay = 1000 / fps;
			ss << fps;
			hThread = CreateThread(NULL, 0, ThreadFun, NULL, 0, NULL);
			hThread2 = CreateThread(NULL, 0, WarningFun, NULL, 0, NULL);
			startupTime = time(NULL);
		}
		if (drawbox_flag == EVENT_LBUTTONDOWN)
		{
			rectangle(frameshow, box, Scalar(255, 0, 0));
		}
		else if (drawbox_flag == EVENT_LBUTTONUP)
		{
			rectangle(frameshow, roiRect, Scalar(255, 0, 0));
		}
		//��ʼ������ģ
		SetEvent(hGetFrameEvent);

		//imshow("FG Mask KNN", fgMaskKNN);
		//imshow("FG Mask MOG 2", fgMaskMOG2);
		//keyboard = waitKey(delay);
		if (device == FILE)
		{
			keyboard = waitKey(33);
		}
		else
		{
			keyboard = waitKey(5);

		}
		totalTime = time(NULL) - startupTime;
		//��ʼ��������
		if (!onsubtract_flag && totalTime>10)
		{
			onsubtract_flag = true;
		}
	}
	TerminateThread(hThread,0);
	destroyAllWindows();
	return EXIT_SUCCESS;
}