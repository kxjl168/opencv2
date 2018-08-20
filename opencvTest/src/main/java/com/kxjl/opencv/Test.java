package com.kxjl.opencv;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
/**
 * 
 * @param map
 * @return
 * @author zj
 * @date 2018-8-4
 */

public class Test {

	static {
		// load the native OpenCV library
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
	}

	public static void main(String[] args) {

		String file = "F:\\IMG\\xiaofang.png";
		String fileout = "F:\\IMG\\xiaofang_gray.png";
		// read the image in gray scale
		Mat image = Imgcodecs.imread(file,
				Imgcodecs.CV_IMWRITE_PAM_FORMAT_RGB_ALPHA);

		Mat grayImage = new Mat();
		Imgproc.cvtColor(image, grayImage, Imgproc.COLOR_BGR2GRAY);

		Imgcodecs.imwrite(fileout, grayImage);

	}

}
