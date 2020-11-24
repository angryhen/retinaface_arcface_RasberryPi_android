#include <jni.h>
#include <string>
#include <opencv2/opencv.hpp>
#include <ctime>

#include <android/asset_manager_jni.h>
#include <android/log.h>
#include "cJSON.h"

// ncnn
#include "net.h"
#include "benchmark.h"

// model
#include "livefacereco.hpp"
#include "live.h"
#include "retinaface.h"
#include "FacePreprocess.h"

#define PI 3.14159265
static ncnn::UnlockedPoolAllocator g_blob_pool_allocator;
static ncnn::PoolAllocator g_workspace_pool_allocator;

static Live liveFace;
class Arcface arcface;


JNIEXPORT jint JNI_OnLoad(JavaVM *vm, void *reserved) {
    __android_log_print(ANDROID_LOG_DEBUG, "FaceRecogNcnn", "JNI_OnLoad");

    ncnn::create_gpu_instance();

    return JNI_VERSION_1_4;
}

JNIEXPORT void JNI_OnUnload(JavaVM *vm, void *reserved) {
    __android_log_print(ANDROID_LOG_DEBUG, "FaceRecogNcnn", "JNI_OnUnload");

    ncnn::destroy_gpu_instance();
}

/*----------------------------------------------------------api list-----------------------------------------------------------------------------*/
/*
	detected = 0:normal image file that includes faces
	detected = 1:face image that only includes single face

	type = 0:all faces
	type = 1:max face

    distance < 1:same person or not

*/
/**
 * 加载模型
 */
extern "C" JNIEXPORT jboolean JNICALL
Java_com_example_testopencv_FaceRecog_Init(JNIEnv *env, jobject thiz, jobject assetManager)
{
    AAssetManager *mgr = AAssetManager_fromJava(env, assetManager);

    //Live detection configs
    struct ModelConfig config1 = {2.7f, 0.0f, 0.0f, 80, 80, "model_1", false};
    struct ModelConfig config2 = {4.0f, 0.0f, 0.0f, 80, 80, "model_2", false};
    vector<struct ModelConfig> configs;
    configs.emplace_back(config1);
    configs.emplace_back(config2);
    liveFace.LoadModel(configs, mgr);
    arcface.LoadModel(mgr);
    return jint(1);
}

/**
 * 提取人脸特征
 */
//extern "C" JNIEXPORT jstring JNICALL
//Java_com_example_testopencv_FaceRecog_Detect(JNIEnv *env, jobject thiz, jobject assetManager, jobject bitmap,
//                                             jboolean use_gpu) {
//    AAssetManager *mgr = AAssetManager_fromJava(env, assetManager);
//    if (use_gpu == JNI_TRUE && ncnn::get_gpu_count() == 0) {
//        return env->NewStringUTF("no vulkan capable gpu");
//    }
//
//    double start_time = ncnn::get_current_time();
//
//    AndroidBitmapInfo info;
//    AndroidBitmap_getInfo(env, bitmap, &info);
//    int width = info.width;
//    int height = info.height;
//    if (width != 320 || height != 240)
//        return NULL;
//    if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888)
//        return NULL;
//
//    // ncnn from bitmap --> opencv Mat
//    ncnn::Mat in = ncnn::Mat::from_android_bitmap(env, bitmap, ncnn::Mat::PIXEL_BGR2RGB);
//
//
//    cv::Mat img(in.h, in.w, CV_8UC3);
//    in.to_pixels(img.data, ncnn::Mat::PIXEL_RGB);
//
//    // loading faces
//    Mat faces;
//    vector<cv::Mat> fc1;
//    std::string pattern_jpg = "/sdcard/DCIM/arcface/20.jpg";
//    std::vector<cv::String> image_names;
//
////    cv::glob(pattern_jpg, image_names);
//    int image_number=image_names.size();
//    if (image_number == 0) {
//        std::cout << "No image files[jpg]" << std::endl;
////        return NULL;
//    }
//    cout <<"loading pictures..."<<endl;
//
//    // prepare param
//    float confidence;
//    vector<float> fps;
//    static double current;
//    static char string[10];
//    static char string1[10];
//    char buff[10];
//    Mat frame;
//    Mat result_cnn;
//
//    // gt face landmark
//    float v1[5][2] = {
//            {30.2946f, 51.6963f},
//            {65.5318f, 51.5014f},
//            {48.0252f, 71.7366f},
//            {33.5493f, 92.3655f},
//            {62.7299f, 92.2041f}};
//
//    cv::Mat src(5, 2, CV_32FC1, v1);
//    memcpy(src.data, v1, 2 * 5 * sizeof(float));
//    double score, angle;
//
//    // retinaface detect
//    auto t = (double) cv::getTickCount();
//    cv::flip (img,img,1);
//
//    std::vector<FaceObject> faceobjects;
//    auto time0 = static_cast<double>(getTickCount());
//
//    // retinaface recog
//    detect_retinaface(img, faceobjects, mgr);
//
//    time0 = ((double)getTickCount() - time0) / getTickFrequency();
//    __android_log_print(ANDROID_LOG_DEBUG, "number of face: ", "%d   detect", jint(faceobjects.size()));
//
////    find the laggest face
//    //obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height
//    int lagerest_face=0, largest_number=0;
//    for (int i = 0; i < faceobjects.size(); i++){
//        int y_ = (int) faceobjects[i].rect.y;
//        int h_ = (int) faceobjects[i].rect.height;
//        if (h_ > lagerest_face){
//            lagerest_face = h_;
//            largest_number = i;
//        }
//     }
//
//    int start_la,end_la;
//    if (faceobjects.size()==0) {
//        start_la= 0;
//        end_la= 0;
//    }
//    else if(largest_face_only){
//        start_la= largest_number;
//        end_la= largest_number+1;
//    }
//    else {
//        start_la=0;
//        end_la= faceobjects.size();
//    }
//
//    //the faces to operate
//    for (int i = start_la; i <end_la; i++) {
//        float x_  = faceobjects[i].rect.x;
//        float y_  = faceobjects[i].rect.y;
//        float x2_ = faceobjects[i].rect.x + faceobjects[i].rect.width;
//        float y2_ = faceobjects[i].rect.y + faceobjects[i].rect.height;
//        int x = (int) x_ ;
//        int y = (int) y_;
//        int x2 = (int) x2_;
//        int y2 = (int) y2_;
//        cout << x << "" << y << "" << x2 << "" << y2 << "" <<endl;
//        struct LiveFaceBox  live_box={x_,y_,x2_,y2_} ;
//
//        // Perspective Transformation
//        float v2[5][2] ={
//            {faceobjects[i].landmark[0].x, faceobjects[i].landmark[0].y},
//            {faceobjects[i].landmark[1].x, faceobjects[i].landmark[1].y},
//            {faceobjects[i].landmark[2].x, faceobjects[i].landmark[2].y},
//            {faceobjects[i].landmark[3].x, faceobjects[i].landmark[3].y},
//            {faceobjects[i].landmark[4].x, faceobjects[i].landmark[4].y},
//        };
//        // compute the turning angle
////        angle = count_angle(v2);
////
////        static std::string hi_name;
////        static std::string liveface;
////        static int stranger,close_enough;
////
/////****************************jump*****************************************************/
////        cv::Mat dst(5, 2, CV_32FC1, v2);
////        memcpy(dst.data, v2, 2 * 5 * sizeof(float));
////
////        // ������
////        cv::Mat m = FacePreprocess::similarTransform(dst, src);
////        cv::Mat aligned = img.clone();
////        cv::warpPerspective(img, aligned, m, cv::Size(96, 112), INTER_LINEAR);
////        resize(aligned, aligned, Size(112, 112), 0, 0, INTER_LINEAR);
////
////        //features of camera image
////        cv::Mat fc2 = arcface.getFeature(aligned);
////
////        // normalize
////        fc2 = Zscore(fc2);
//
////        //the similarity score
////        vector<double> score_;
////        for (unsigned int compare_ = 0; compare_ < image_number; ++ compare_){
////            score_.push_back(CosineDistance(fc1[compare_], fc2));
////        }
////        int maxPosition = max_element(score_.begin(),score_.end()) - score_.begin();
////        current=score_[maxPosition];
////        score_.clear();
////        sprintf(string, "%.4f", current);
////
////        if (current >= face_thre && y2-y>= distance_threshold){
////            //put name
////            int slant_position=image_names[maxPosition].rfind ('/');
////            cv::String name = image_names[maxPosition].erase(0,slant_position+1);
////            name=name.erase( name.length()-4, name.length()-1);
////            hi_name="Hi,"+name;
////            putText(result_cnn, hi_name, cv::Point(5, 60), cv::FONT_HERSHEY_SIMPLEX,0.75, cv::Scalar(0, 255, 255),2);
////            cout<<name<<endl;
////            //determin whethe it is a fake face
////            confidence=liveFace.Detect(frame,live_box);
////
////            sprintf(string1, "%.4f", confidence);
////            cv::putText(result_cnn,string1, Point(x*ratio_x, y2*ratio_y+20), cv::FONT_HERSHEY_SIMPLEX,0.75, cv::Scalar(0,255,255),2);
////            if (confidence<=true_thre){
////                putText(result_cnn, "Fake face!!", cv::Point(5, 80), cv::FONT_HERSHEY_SIMPLEX,0.75, cv::Scalar(0, 0, 255),2);
////                liveface="Fake face!!";
////            }
////            else{
////                putText(result_cnn, "True face", cv::Point(5, 80), cv::FONT_HERSHEY_SIMPLEX,0.75, cv::Scalar(255, 255, 0),2);
////                liveface="True face";
////            }
////            cout<<liveface<<endl;
////            stranger=0;
////            close_enough=1;
////        }
////
////        else if(current >= face_thre && y2-y < distance_threshold){
////                //put name
////                int slant_position=image_names[maxPosition].rfind ('/');
////                cv::String name = image_names[maxPosition].erase(0,slant_position+1);
////                name=name.erase( name.length()-4, name.length()-1);
////                hi_name="Hi,"+name;
////                putText(result_cnn, hi_name, cv::Point(5, 60), cv::FONT_HERSHEY_SIMPLEX,0.75, cv::Scalar(0, 255, 255),2);
////                //Ask be closer to avoid mis-reco
////                putText(result_cnn, "Closer please", cv::Point(5, 80), cv::FONT_HERSHEY_SIMPLEX,0.75, cv::Scalar(0, 255, 255),2);
////                cout<<"Closer please"<<endl;
////                stranger=0;
////                close_enough=0;
////            }
////            else {
////                        putText(result_cnn, "Stranger", cv::Point(5, 60), cv::FONT_HERSHEY_SIMPLEX,0.75, cv::Scalar(0, 0, 255),2);
////                        cout<<"Stranger"<<endl;
////                        stranger=1;
////            }
////         //highlight the significant landmarks on face
////        for (int j = 0; j < 5; j += 1) {
////            if (j == 0 or j == 3) {
////                cv::circle(result_cnn, faceobjects[i].landmark[j], 3,
////                        Scalar(0, 255, 0),
////                        FILLED, LINE_AA);
////            } else if (j==2){
////                cv::circle(result_cnn, faceobjects[i].landmark[j], 3,
////                        Scalar(255, 0, 0),
////                        FILLED, LINE_AA);
////            }
////                else {
////                cv::circle(result_cnn, faceobjects[i].landmark[j], 3,
////                        Scalar(0, 0, 255),
////                        FILLED, LINE_AA);
////                }
////        }
//        }
//
//    char tmp[32];
//
//    sprintf(tmp, "%.3f", time0);
//    std::string result_str = std::string("use time") + " = " + tmp;
//    jstring result = env->NewStringUTF(result_str.c_str());
//    return result;
//    }


extern "C" JNIEXPORT jobjectArray JNICALL
Java_com_example_testopencv_FaceRecog_Detect(JNIEnv* env, jobject thiz, jobject bitmap, jboolean use_gpu, jobject assetManager) {
    AAssetManager *mgr = AAssetManager_fromJava(env, assetManager);
    auto box_cls = env->FindClass("com/example/testopencv/Box");
    auto cid = env->GetMethodID(box_cls, "<init>", "(FFFFIFFFFFFFFFFF)V");

    double start_time = ncnn::get_current_time();

    AndroidBitmapInfo info;
    AndroidBitmap_getInfo(env, bitmap, &info);
    int width = info.width;
    int height = info.height;
    if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888)
        return NULL;

//    cv::Mat rgbImg = cv::imread("/sdcard/DCIM/Camera/IMG_20200915_161302.jpg");
//    cv::Mat rgbImg2 = cv::imread("/mnt/sdcard/DCIM/Camera/IMG_20200915_161302.jpg");

    // ncnn from bitmap
    // ncnn::Mat in = ncnn::Mat::from_android_bitmap_resize(env, bitmap, ncnn::Mat::PIXEL_RGB, 320, 240);
    ncnn::Mat in = ncnn::Mat::from_android_bitmap(env, bitmap, ncnn::Mat::PIXEL_BGR2RGB);

    cv::Mat img(in.h, in.w, CV_8UC3);
    in.to_pixels(img.data, ncnn::Mat::PIXEL_RGB);

    // loading faces
    Mat faces;
    vector<cv::Mat> fc1;
    std::string pattern_jpg = "/sdcard/DCIM/arcface/20.jpg";
    std::vector<cv::String> image_names;

//    cv::glob(pattern_jpg, image_names);
    int image_number = image_names.size();
    if (image_number == 0) {
        std::cout << "No image files[jpg]" << std::endl;
//        return NULL;
    }
    cout << "loading pictures..." << endl;

    // prepare param
    float confidence;
    vector<float> fps;
    static double current;
    static char string[10];
    static char string1[10];
    char buff[10];
    Mat frame;
    Mat result_cnn;

    // gt face landmark
    float v1[5][2] = {
            {30.2946f, 51.6963f},
            {65.5318f, 51.5014f},
            {48.0252f, 71.7366f},
            {33.5493f, 92.3655f},
            {62.7299f, 92.2041f}};

    cv::Mat src(5, 2, CV_32FC1, v1);
    memcpy(src.data, v1, 2 * 5 * sizeof(float));
    double score, angle;

    // retinaface detect
    auto t = (double) cv::getTickCount();
//    cv::flip(img, img, 1);

    std::vector<FaceObject> faceobjects;
    auto time0 = static_cast<double>(getTickCount());
    // retinaface recog
    detect_retinaface(img, faceobjects, mgr);
    __android_log_print(ANDROID_LOG_DEBUG, "number of face: ", "%d   detect",
                        jint(faceobjects.size()));
    time0 = ((double) getTickCount() - time0) / getTickFrequency();


    // find the laggest face
    // obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height
    int lagerest_face = 0, largest_number = 0;
    for (int i = 0; i < faceobjects.size(); i++) {
        int y_ = (int) faceobjects[i].rect.y;
        int h_ = (int) faceobjects[i].rect.height;
        if (h_ > lagerest_face) {
            lagerest_face = h_;
            largest_number = i;
        }
    }

    int start_la, end_la;
    if (faceobjects.size() == 0) {
        start_la = 0;
        end_la = 0;
    } else if (largest_face_only) {
        start_la = largest_number;
        end_la = largest_number + 1;
    } else {
        start_la = 0;
        end_la = faceobjects.size();
    }

    float x = 0;
    float y = 0;
    float x2 = 0;
    float y2 = 0;
    float v2[5][2];
    //the faces to operate
    for (int i = start_la; i < end_la; i++) {
        float x_ = faceobjects[i].rect.x;
        float y_ = faceobjects[i].rect.y;
        float x2_ = faceobjects[i].rect.x + faceobjects[i].rect.width;
        float y2_ = faceobjects[i].rect.y + faceobjects[i].rect.height;
        x = x_;
        y = y_;
        x2 = x2_;
        y2 = y2_;
        cout << x << "" << y << "" << x2 << "" << y2 << "" << endl;
        struct LiveFaceBox live_box = {x_, y_, x2_, y2_};

        // Perspective Transformation
        float v2[5][2] = {
                {faceobjects[i].landmark[0].x, faceobjects[i].landmark[0].y},
                {faceobjects[i].landmark[1].x, faceobjects[i].landmark[1].y},
                {faceobjects[i].landmark[2].x, faceobjects[i].landmark[2].y},
                {faceobjects[i].landmark[3].x, faceobjects[i].landmark[3].y},
                {faceobjects[i].landmark[4].x, faceobjects[i].landmark[4].y},
        };
        // *************************************************************************************************
        angle = count_angle(v2);
        static std::string hi_name;
        static std::string liveface;
        static int stranger, close_enough;
        cv::Mat dst(5, 2, CV_32FC1, v2);
        memcpy(dst.data, v2, 2 * 5 * sizeof(float));

        // arcFace recog
        cv::Mat m = FacePreprocess::similarTransform(dst, src);
        cv::Mat aligned = img.clone();
        cv::warpPerspective(img, aligned, m, cv::Size(96, 112), INTER_LINEAR);
        resize(aligned, aligned, Size(112, 112), 0, 0, INTER_LINEAR);
        //features of camera image
        cv::Mat fc2 = arcface.getFeature(aligned);
        // normalize
        fc2 = Zscore(fc2);

        // live detect
        confidence=liveFace.Detect(img,live_box);
        __android_log_print(ANDROID_LOG_DEBUG, "Live detect: ", "%.4f", confidence);
        // *************************************************************************************************
    }

    jobjectArray ret = env->NewObjectArray(faceobjects.size(), box_cls, NULL);
    for (size_t i = 0; i < faceobjects.size(); i++) {
        env->PushLocalFrame(1);
        jobject obj;
        obj = env->NewObject(box_cls, cid,
                             faceobjects[i].rect.x,
                             faceobjects[i].rect.y,
                             faceobjects[i].rect.x + faceobjects[i].rect.width,
                             faceobjects[i].rect.y + faceobjects[i].rect.height,
                             0,
                             float(0),
                             faceobjects[i].landmark[0].x, faceobjects[i].landmark[0].y,
                             faceobjects[i].landmark[1].x, faceobjects[i].landmark[1].y,
                             faceobjects[i].landmark[2].x, faceobjects[i].landmark[2].y,
                             faceobjects[i].landmark[3].x, faceobjects[i].landmark[3].y,
                             faceobjects[i].landmark[4].x, faceobjects[i].landmark[4].y);
        obj = env->PopLocalFrame(obj);
        env->SetObjectArrayElement(ret, i, obj);
    }
    return ret;
}
