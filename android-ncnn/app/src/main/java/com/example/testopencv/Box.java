package com.example.testopencv;

import android.graphics.Color;
import android.graphics.RectF;

import java.util.Random;

//public class Box
//{
//    public float x;
//    public float y;
//    public float w;
//    public float h;
//    public String label;
//    public float prob;
//}

public class Box {
    public float x0;
    public float y0;
    public float x1;
    public float y1;
    public float v[][] = new float[5][2];
    private int label;
    private final float score;
    private static final String[] labels={"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
            "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
            "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
            "hair drier", "toothbrush"};
    public Box(float x0, float y0, float x1, float y1, int label, float score,
               float v00, float v01, float v10, float v11, float v20, float v21, float v30, float v31, float v40, float v41){
        this.x0 = x0;
        this.y0 = y0;
        this.x1 = x1;
        this.y1 = y1;
        this.label = label;
        this.score = score;

        float[][] a = {
                {v00,v01},
                {v10,v11},
                {v20,v21},
                {v30,v31},
                {v40,v41}};
        System.arraycopy(a,0,this.v,0,a.length); //通过arraycopy()函数拷贝数组
    }


    public RectF getRect(){
        return new RectF(x0,y0,x1,y1);
    }

    public String getLabel(){
        String face = "face";
//        return labels[label];
        return face;
    }

    public float getScore(){
        float score = 1;
        return score;
    }

    public int getColor(){
        Random random = new Random(label);
        return Color.argb(255,random.nextInt(256),random.nextInt(256),random.nextInt(256));
    }
}
