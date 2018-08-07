package com.kxjl.opencv;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ScheduledExecutorService;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.RotatedRect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Dnn;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.objdetect.Objdetect;
import org.opencv.video.Video;
import org.opencv.videoio.VideoCapture;

import com.sun.scenario.effect.light.Light;

import javafx.beans.property.ObjectProperty;
import javafx.beans.property.SimpleObjectProperty;
import javafx.fxml.FXML;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.Slider;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.stage.FileChooser;
import javafx.stage.Stage;

/**
 * The controller associated to the only view of our application. The
 * application logic is implemented here. It handles the button for opening an
 * image and perform all the operation related to the Fourier transformation and
 * antitransformation.
 * 
 * @author <a href="mailto:luigi.derussis@polito.it">Luigi De Russis</a>
 * @version 2.0 (2017-03-10)
 * @since 1.0 (2013-12-11)
 * 
 */
public class FourierController {
	// images to show in the view
	@FXML
	private ImageView originalImage;
	@FXML
	private ImageView transformedImage;
	@FXML
	private ImageView antitransformedImage;

	@FXML
	private ImageView transformedImage2;
	@FXML
	private ImageView antitransformedImage2;
	// a FXML button for performing the transformation
	@FXML
	private Button transformButton;
	// a FXML button for performing the antitransformation
	@FXML
	private Button antitransformButton;

	@FXML
	private Label hsvCurrentValues;
	// property for object binding
	private ObjectProperty<String> hsvValuesProp;

	@FXML
	private Slider hueStart;
	@FXML
	private Slider hueStop;
	@FXML
	private Slider saturationStart;
	@FXML
	private Slider saturationStop;
	@FXML
	private Slider valueStart;
	@FXML
	private Slider valueStop;

	// the main stage
	private Stage stage;
	// the JavaFX file chooser
	private FileChooser fileChooser;
	// support variables
	private Mat image;

	private Mat grayimage;
	private List<Mat> planes;
	// the final complex image
	private Mat complexImage;

	// face cascade classifier
	private CascadeClassifier faceCascade;
	private int absoluteFaceSize;

	@FXML
	private Slider threshold;

	/**
	 * Init the needed variables
	 */
	protected void init() {
		this.fileChooser = new FileChooser();
		this.image = new Mat();
		this.grayimage = new Mat();

		hsvValuesProp = new SimpleObjectProperty<>();
		this.hsvCurrentValues.textProperty().bind(hsvValuesProp);

		// face
		this.faceCascade = new CascadeClassifier();
		this.absoluteFaceSize = 0;

		this.faceCascade.load("resources/lbpcascades/lbpcascade_frontalface.xml");

		// this.planes = new ArrayList<>();
		// this.complexImage = new Mat();
	}

	private double getHistAverage(Mat hsvImg, Mat hueValues) {
		// init
		double average = 0.0;
		Mat hist_hue = new Mat();
		// 0-180: range of Hue values
		MatOfInt histSize = new MatOfInt(180);
		List<Mat> hue = new ArrayList<>();
		hue.add(hueValues);

		// compute the histogram
		Imgproc.calcHist(hue, new MatOfInt(0), new Mat(), hist_hue, histSize, new MatOfFloat(0, 179));

		// get the average Hue value of the image
		// (sum(bin(h)*h))/(image-height*image-width)
		// -----------------
		// equivalent to get the hue of each pixel in the image, add them, and
		// divide for the image size (height and width)
		for (int h = 0; h < 180; h++) {
			// for each bin, get its value and multiply it for the corresponding
			// hue
			average += (hist_hue.get(h, 0)[0] * h);
		}

		// return the average hue of the image
		double av = average / hsvImg.size().height / hsvImg.size().width;
		System.err.println("av:" + av);
		return av;
	}

	/**
	 * 输入 gray图像
	 * 
	 * @param frame
	 * @return
	 * @author zj
	 * @date 2018年8月6日
	 */
	private Mat BackgroundRemoval(Mat frame) {

		// init
		Mat hsvImg = new Mat();
		List<Mat> hsvPlanes = new ArrayList<>();
		Mat thresholdImg = new Mat();

		/*
		 * if (this.inverse.isSelected()) thresh_type = Imgproc.THRESH_BINARY;
		 */

		// threshold the image with the average hue value

		hsvImg.create(frame.size(), CvType.CV_8U);
		Mat ori = new Mat();
		Imgproc.cvtColor(frame, ori, Imgproc.COLOR_GRAY2BGR);
		Imgproc.cvtColor(ori, hsvImg, Imgproc.COLOR_BGR2HSV);

		Core.split(hsvImg, hsvPlanes);
		double threshValue = this.getHistAverage(hsvImg, hsvPlanes.get(0));

		double threshValue2 = getthresh(frame);
		System.out.println("av threshValue:" + threshValue);
		System.out.println("a2v threshValue:" + threshValue2);
		/*
		 * double threshValue = this.threshold.getValue();// this.getHistAverage(hsvImg,
		 * hsvPlanes.get(0));
		 * 
		 * double to=threshold2.getValue();
		 * 
		 * int s3=(int)threshold3.getValue(); int s4=(int)threshold4.getValue();
		 */

		// threshValue2;//
		threshValue = (int) this.threshold.getValue(); // 140;//
		// this.getHistAverage(hsvImg, hsvPlanes.get(0));

		System.out.println("threshValue:" + threshValue);
		double to = 255;// threshold2.getValue(); //10;//
		System.out.println("to:" + to);

		int s3 = 1;// (int)threshold3.getValue();
		int s4 = 1;// (int)threshold4.getValue();

		int thresh_type = Imgproc.THRESH_BINARY_INV;
		// threshValue
		// 双值化图像
		Imgproc.threshold(frame, thresholdImg, threshValue, to, thresh_type);

		Imgproc.blur(thresholdImg, thresholdImg, new Size(3, 3));

		// dilate to fill gaps, erode to smooth edges
		Imgproc.dilate(thresholdImg, thresholdImg, new Mat(), new Point(-1, -1), s3);
		Imgproc.erode(thresholdImg, thresholdImg, new Mat(), new Point(-1, -1), s4);

		Imgproc.threshold(thresholdImg, thresholdImg, threshValue, to, thresh_type);

		// create the new image
		Mat foreground = new Mat(frame.size(), CvType.CV_8UC3, new Scalar(255, 255, 255));
		thresholdImg.copyTo(foreground);// , thresholdImg);

		// show the current selected HSV range
		String valuesToPrint = "threshValue: " + threshValue + "- to：" + to + " s3:" + s3 + "-  s4：" + s4;
		Utils.onFXThread(this.hsvValuesProp, valuesToPrint);

		return foreground;
	}

	@FXML
	private void doBackgroundRemoval() {

		Mat src2 = new Mat();
		this.grayimage.copyTo(src2);
		// Mat blurredImage = new Mat();
		// Imgproc.blur(src2, blurredImage, new Size(1, 1));
		Mat matc = new Mat();
		// Imgproc.cvtColor(src2, matc, Imgproc.COLOR_GRAY2BGR);

		avLight(this.image);

		String path = FourierController.class.getResource("/").getPath();
		File f = new File(path + "\\out");
		if (!f.exists())
			f.mkdirs();

		int imgwidth = src2.width();
		int imgheight = src2.height();

		matc = BackgroundRemoval(src2);

		// findline(matc);

		// findContours(matc);
		//

		this.updateImageView(antitransformedImage, Utils.mat2Image(matc)); // set a
		this.antitransformedImage.setFitHeight(350); // preserve image ratio

		// dorect(matc);

	}

	private void findContours(Mat cmat) {

		List<MatOfPoint> contours = new ArrayList<>();
		Mat hierarchy = new Mat();
		// find contours
		Imgproc.findContours(cmat, contours, hierarchy, Imgproc.RETR_CCOMP, Imgproc.CHAIN_APPROX_SIMPLE);

		Mat frame = new Mat();
		this.image.copyTo(frame);
		// if any contour exist...
		if (hierarchy.size().height > 0 && hierarchy.size().width > 0) { // for each contour, display it in blue
			for (int idx = 0; idx >= 0; idx = (int) hierarchy.get(0, idx)[0]) {
				Imgproc.drawContours(frame, contours, idx, new Scalar(250, 0, 0));
			}
		}

		this.updateImageView(antitransformedImage, Utils.mat2Image(frame)); // set a
		this.antitransformedImage.setFitHeight(350); // preserve image ratio

	}

	private void findline(Mat mask) {

		Mat matc = new Mat();
		this.image.copyTo(matc);

		Mat line = new Mat();
		Imgproc.HoughLinesP(mask, line, 1, Math.PI / 180, 100, 85, 20);

		List<Point> pts = new ArrayList<>();
		Map<Double, Integer> xnums = new HashMap<>();
		Map<Double, Integer> x2nums = new HashMap<>();
		for (int i = 0; i < line.rows(); i++) {

			try {

				Point ps = new Point(line.get(i, 0)[0], line.get(i, 0)[1]);
				Point pend = new Point(line.get(i, 0)[2], line.get(i, 0)[3]);

				double x1 = line.get(i, 0)[0];
				double x2 = line.get(i, 0)[2];

				int num = xnums.get(x1) == null ? 0 : xnums.get(x1);
				xnums.put(x1, num + 1);

				int num2 = x2nums.get(x2) == null ? 0 : x2nums.get(x2);
				x2nums.put(x2, num2 + 1);

				pts.add(ps);
				pts.add(pend);

				// Imgproc.line(matcgray, ps, pend, new Scalar(0, 0, 0), 1);

				Imgproc.line(matc, ps, pend, new Scalar(0, 0, 255), 1);
				Imgproc.drawMarker(matc, ps, new Scalar(0, 255, 0), Imgproc.MARKER_DIAMOND, 20, 5, 8);
				Imgproc.drawMarker(matc, pend, new Scalar(255, 0, 0), Imgproc.MARKER_SQUARE, 20, 5, 8);
			} catch (Exception e) {
				e.printStackTrace();
			}
		}

		this.updateImageView(antitransformedImage, Utils.mat2Image(matc)); // set a
		this.antitransformedImage.setFitHeight(350); // preserve image ratio

	}

	/**
	 * 腐蚀膨胀
	 * 
	 * @author zj
	 * @date 2018年8月6日
	 */
	@FXML
	private void tr2dialate_erode() {
		fadnp(this.image);
	}

	private void fadnp(Mat input) {
		Mat nmat = new Mat();
		input.copyTo(nmat);

		// 腐蚀/膨胀
		Mat m_dilate = new Mat();
		Mat structElement1 = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(4, 4), new Point(-1, -1));
		Imgproc.dilate(nmat, m_dilate, structElement1);
		// imshow("膨胀", out1);imwrite("膨胀.jpg", out1);

		// 腐蚀,去除离散点
		Mat m_erode = new Mat();
		Imgproc.erode(nmat, m_erode, structElement1);

		// show the image
		this.updateImageView(transformedImage, Utils.mat2Image(m_erode));

		this.updateImageView(antitransformedImage, Utils.mat2Image(m_dilate));
		// set a fixed width
		this.transformedImage.setFitWidth(450);
		this.antitransformedImage.setFitWidth(450);

	}

	@FXML
	protected void tr2gary() {

		// show the image
		this.updateImageView(transformedImage, Utils.mat2Image(this.grayimage));
		// set a fixed width
		this.transformedImage.setFitWidth(450);
		// this.transformedImage.setFitHeight(450);

	}

	@FXML
	protected void tr2candy() {

		Mat nmat = new Mat();
		this.image.copyTo(nmat);

		Mat t = doCanny(nmat);

		// show the image
		this.updateImageView(antitransformedImage, Utils.mat2Image(t));

		// set a fixed width
		this.antitransformedImage.setFitWidth(450);
		// this.transformedImage.setFitHeight(450);

	}

	@FXML
	protected void t2ruihua() {
		Mat nmat = new Mat();
		this.image.copyTo(nmat);

		Mat t = ruihua(nmat);

		// show the image
		this.updateImageView(antitransformedImage, Utils.mat2Image(t));

		// set a fixed width
		this.antitransformedImage.setFitWidth(450);
	}

	@FXML
	private void removeBackHsv() {

		Mat nmat = new Mat();
		this.grayimage.copyTo(nmat);

		Imgproc.cvtColor(nmat, nmat, Imgproc.COLOR_GRAY2BGR);

		int delval = avLight(nmat);
		// delval=172;
		// nmat = getSimpleImgForJashiz(nmat, 0, 150);

		nmat = dohsvRemoveBack(nmat);

		this.updateImageView(this.antitransformedImage, Utils.mat2Image(nmat));
		this.antitransformedImage.setFitWidth(450);

	}

	/**
	 * 获取给定BGR图像的平均亮度
	 * 
	 * @param matc
	 * @return
	 * @author zj
	 * @date 2018年8月2日
	 */
	private int avLight(Mat matc) {

		Mat tmp = new Mat();
		matc.copyTo(tmp, matc);
		// 计算图片平均亮度
		Mat hsvImage = new Mat();

		Mat gary = new Mat();
		Imgproc.cvtColor(matc, gary, Imgproc.COLOR_BGR2GRAY);

		hsvImage = new Mat();
		Imgproc.cvtColor(tmp, hsvImage, Imgproc.COLOR_BGR2HLS);

		int imgheight = hsvImage.height();
		int imgwidth = hsvImage.width();

		for (int i = 0; i < imgwidth - 1; i++) {
			for (int j = 0; j < imgheight; j++) {
				// int j=imgheight/2;
				double[] vals = gary.get(j, i);
				double light = vals[0];

				if (i < 10) {
					if (j == imgheight / 2) {
						System.out.print(light + " ");
					}
				}

			}

		}

		System.out.println("--------------------");

		// Imgcodecs.imwrite(f.getAbsolutePath() + "\\" + "hls.jpg", hsvImage);
		double total = 0;
		double avlight = 0;
		for (int i = 0; i < imgwidth - 1; i++) {
			for (int j = 0; j < imgheight; j++) {
				// int j=imgheight/2;
				double[] vals = hsvImage.get(j, i);
				double light = vals[1];
				total += light;

				if (i < 10) {
					if (j == imgheight / 2) {
						System.out.print(light + " ");
					}
				}

			}

		}
		avlight = total / ((double) (imgwidth * imgheight));

		System.out.println("平均亮度:" + avlight);

		int delval = 90;
		if (avlight <= 160)
			delval = 85;
		else if (avlight > 160 && avlight <= 170)
			delval = 91;
		else if (avlight > 170 && avlight <= 190)
			delval = 141;
		else if (avlight > 190)
			delval = 141;

		return delval;
	}

	/**
	 * 去除图片背景，突出文字 ，hsv调整，
	 * 
	 * @param frame
	 * @author zj
	 * @date 2018年8月2日
	 */
	private Mat getSimpleImgForJashiz(Mat frame, double starth, double endh) {

		Mat mask = new Mat();

		Scalar minValues = new Scalar(starth, this.saturationStart.getValue(), this.valueStart.getValue());
		Scalar maxValues = new Scalar(endh, this.saturationStop.getValue(), this.valueStop.getValue());

		// show the current selected HSV range
		String valuesToPrint = "Hue range: " + minValues.val[0] + "-" + maxValues.val[0] + "\tSaturation range: "
				+ minValues.val[1] + "-" + maxValues.val[1] + "\tValue range: " + minValues.val[2] + "-"
				+ maxValues.val[2];
		Utils.onFXThread(this.hsvValuesProp, valuesToPrint);

		// threshold HSV image to select tennis balls
		Core.inRange(frame, minValues, maxValues, mask);

		return mask;
	}

	private Mat dohsvRemoveBack(Mat input) {
		// init
		Mat blurredImage = new Mat();
		Mat hsvImage = new Mat();
		Mat mask = new Mat();
		Mat morphOutput = new Mat();

		Mat frame = new Mat();
		input.copyTo(frame);

		// remove some noise
		Imgproc.blur(frame, blurredImage, new Size(3, 3));

		// convert the frame to HSV
		//// Imgproc.cvtColor(blurredImage, hsvImage, Imgproc.COLOR_BGR2HSV);
		// Imgproc.cvtColor(blurredImage, hsvImage, Imgproc.COLOR_BGR2HLS);

		// Imgproc.cvtColor(blurredImage, hsvImage, Imgproc.COLOR_BGR2GRAY);

		// Imgproc.cvtColor(hsvImage, hsvImage, Imgproc.COLOR_GRAY2BGR);

		// get thresholding values from the UI
		// remember: H ranges 0-180, S and V range 0-255
		Scalar minValues = new Scalar(this.hueStart.getValue(), this.saturationStart.getValue(),
				this.valueStart.getValue());
		Scalar maxValues = new Scalar(this.hueStop.getValue(), this.saturationStop.getValue(),
				this.valueStop.getValue());

		// show the current selected HSV range
		String valuesToPrint = "Hue range: " + minValues.val[0] + "-" + maxValues.val[0] + "\tSaturation range: "
				+ minValues.val[1] + "-" + maxValues.val[1] + "\tValue range: " + minValues.val[2] + "-"
				+ maxValues.val[2];
		Utils.onFXThread(this.hsvValuesProp, valuesToPrint);

		// threshold HSV image to select tennis balls
		Core.inRange(frame, minValues, maxValues, mask);

		return mask;
	}

	private Mat ruihua(Mat input) {

		Mat kernel = new Mat(3, 3, CvType.CV_32F);
		// int[] values = {0, -1, 0, -1, 5, -1, 0, -1, 0};
		// Log.d("imageType", CvType.typeToString(src.type()) + "");
		kernel.put(0, 0, 0, -1, 0, -1, 5, -1, 0, -1, 0);
		// Imgproc.filter2D(src, src, src_gray.depth(), kernel);

		// Mat kern = new Mat(3, 3,CvType.CV_8UC3,new bytebu
		// -1, 5, -1,
		// [0, -1, 0]});

		// Mat kern = new Mat();//
		// (3, 0,3,new Scalar({-1 ,0,
		// -1, 5, -1,
		// [0, -1, 0]});
		Mat dstImage = new Mat();
		Imgproc.filter2D(input, dstImage, input.depth(), kernel);
		return dstImage;
	}

	@FXML
	private void t2blur() {

		Mat frame = new Mat();

		Mat dest = new Mat();
		this.image.copyTo(frame);

		// ģ���뾶3*3
		Imgproc.blur(frame, dest, new Size(2, 2));

		this.updateImageView(this.transformedImage, Utils.mat2Image(dest));
		this.transformedImage.setFitWidth(450);
	}

	@FXML
	private void t2big() {

		Mat frame = new Mat();

		Mat dest = new Mat();
		Mat dest2 = new Mat();
		this.image.copyTo(frame);

		// �Ŵ�
		Imgproc.pyrUp(frame, dest, new Size(frame.width() * 2, frame.height() * 2));

		Imgproc.pyrDown(frame, dest2, new Size(frame.width() / 2, frame.height() / 2));

		dest = new Mat(frame, new Rect(0, 0, frame.width() / 2, frame.height() / 2));

		this.updateImageView(this.transformedImage, Utils.mat2Image(dest));
		this.transformedImage.fitWidthProperty();
		this.transformedImage.fitHeightProperty();

		this.updateImageView(this.antitransformedImage, Utils.mat2Image(dest2));
		this.antitransformedImage.fitWidthProperty();
		this.antitransformedImage.fitHeightProperty();
	}

	/**
	 * 获取灰度图的1/3区域高度的平均灰度值
	 * 
	 * @param input
	 *            灰度图
	 * @return
	 * @author zj
	 * @date 2018年8月7日
	 */
	private double getthresh(Mat input) {
		double v = 0;

		Mat gray = new Mat();
		input.copyTo(gray);
		// Imgproc.cvtColor(input, gray, Imgproc.COLOR_BGR2GRAY);

		// //函数功能：直方图均衡化，该函数能归一化图像亮度和增强对比度
		// Imgproc.equalizeHist(gray, gray);

		int imgwidth = gray.width();
		int imgheight = gray.height();
		double total = 0;
		for (int i = 0; i < imgwidth; i++) {
			for (int j = imgheight / 3; j < imgheight * 2 / 3; j++) {
				// int j=imgheight/2;
				double[] vals = gray.get(j, i);
				double light = vals[0];
				total += light;

			}

		}
		v = total / (imgwidth * imgheight / 3);
		System.out.println("v:" + v);
		return v;
	}

	/**
	 * 获取灰度图的1/3区域高度的最大灰度值
	 * 
	 * @param input
	 *            灰度图
	 * @return
	 * @author zj
	 * @date 2018年8月7日
	 */
	private double getMaxthresh(Mat input) {
		double v = 0;

		Mat gray = new Mat();
		input.copyTo(gray);
		// Imgproc.cvtColor(input, gray, Imgproc.COLOR_BGR2GRAY);

		// //函数功能：直方图均衡化，该函数能归一化图像亮度和增强对比度
		// Imgproc.equalizeHist(gray, gray);

		int imgwidth = gray.width();
		int imgheight = gray.height();
		double max = 0;
		for (int i = 0; i < imgwidth; i++) {
			for (int j = imgheight / 3; j < imgheight * 2 / 3; j++) {
				// int j=imgheight/2;
				double[] vals = gray.get(j, i);
				double light = vals[0];
				if (light > max)
					max = light;

			}

		}

		System.out.println("max gray:" + max);
		return max;
	}

	@FXML
	private void cancueIdCard() {

		Mat m = new Mat();
		Mat filtered = new Mat();
		Mat thresholdImg = new Mat();
		Mat dilated_edges = new Mat();

		Mat rectM = new Mat();
		this.grayimage.copyTo(m);
		this.image.copyTo(rectM);

		m.copyTo(filtered);
		// 滤波，模糊处理，消除某些背景干扰信息
		Imgproc.blur(m, filtered, new Size(1, 1));

		// 腐蚀操作，消除某些背景干扰信息
		// Imgproc.erode(filtered, filtered, new Mat(), new Point(-1, -1), 1);// 1, 1);

		// double maxgrayval = getMaxthresh(m);
		double maxgrayval = 40;// getthresh(m);
		System.out.println("maxgrayval:" + maxgrayval);

		// int thresh_type = Imgproc.THRESH_BINARY_INV; //反转
		int thresh_type = Imgproc.THRESH_OTSU;// 前后背景区分
		// threshValue
		// 双值化图像
		double to = 255;
		Imgproc.threshold(filtered, thresholdImg, maxgrayval * 3 / 4, to, thresh_type);

		// Imgproc.blur(thresholdImg, thresholdImg, new Size(3, 3));

		// dilate to fill gaps, erode to smooth edges
		// Imgproc.dilate(thresholdImg, thresholdImg, new Mat(), new Point(-1, -1), 3);
		// Imgproc.erode(thresholdImg, thresholdImg, new Mat(), new Point(-1, -1), 3);

		// Imgproc.threshold(thresholdImg, thresholdImg, maxgrayval * 2 / 3, to,
		// thresh_type);

		savetoImg(thresholdImg, "thresholdImg");

		Rect r = dorect(thresholdImg, rectM);

		savetoImg(rectM, "rectM");

		Mat fMat = new Mat(thresholdImg, r);

		Mat fRMat = new Mat(rectM, r);

		savetoImg(fMat, "gray_rect");
		savetoImg(fRMat, "rectM_rect");

		showImg(this.transformedImage, rectM);

		Mat yhist = new Mat();
		List<Rect> rects = cany(fMat, yhist);

		showImg(this.transformedImage2, yhist);

		Imgproc.cvtColor(fMat, fMat, Imgproc.COLOR_GRAY2BGR);
		for (int i = 0; i < rects.size(); i++) {
			// if(rects.get(i).height>30)
			// {
			System.out.println(i + ": x:" + rects.get(i).x + " y:" + rects.get(i).y + " height:" + rects.get(i).height);
			Imgproc.rectangle(fMat, rects.get(i).tl(), rects.get(i).br(), new Scalar(0, 0, 255));
			// }
		}

		showImg(this.antitransformedImage, fMat);

		/*
		 * Mat xhist = canx(fMat);
		 * 
		 * showImg(this.antitransformedImage2, xhist);
		 */
	}

	/**
	 * 计算xy投影
	 * 
	 * 输入二值化/rect切割好的证件图片
	 * 
	 * @author zj
	 * @date 2018年8月7日
	 */
	private List<Rect> cany(Mat input, Mat output) {

		int imgheight = input.height();
		int imgwidth = input.width();

		Mat hist = new Mat(1, imgheight, CvType.CV_32FC1);

		Core.normalize(hist, hist, 0, hist.rows(), Core.NORM_MINMAX, -1, new Mat());
		double[] v = new double[imgheight];

		for (int j = 0; j < imgheight; j++) {
			v[j] = 0;
		}
		hist.put(0, 0, v);

		double max = imgheight;// 0;

		for (int i = 0; i < imgheight; i++) {
			for (int j = 0; j < imgwidth / 2; j++) {
				if (input.get(i, j)[0] == 0) {

					double num = hist.get(0, i)[0];
					num++;
					hist.put(0, i, num);
				} else {
					int k = 0;
				}
			}
			double num = hist.get(0, i)[0];
			if (num > max)
				max = num;
		}

		Mat hist2 = new Mat(imgheight, imgwidth, CvType.CV_8UC1);

		Core.normalize(hist2, hist2, 0, hist2.rows(), Core.NORM_MINMAX, -1, new Mat());
		double[] v2 = new double[imgwidth * imgheight];

		for (int j = 0; j < imgwidth * imgheight; j++) {
			v2[j] = 255;
		}
		hist2.put(0, 0, v2);
		for (int j = 0; j < imgheight; j++) {

			double y = (hist.get(0, j)[0] / max) * imgwidth;

			// B component or gray image
			Imgproc.line(hist2, new Point(0, j), new Point(y, j), new Scalar(0, 0, 0), 2, 8, 0);
		}

		Imgproc.cvtColor(hist2, hist2, Imgproc.COLOR_GRAY2BGR);

		hist2.copyTo(output);

		// 计算纵向区域
		List<Rect> rects = new ArrayList<>();
		boolean isstart = false;

		double start_val1 = 0;
		double start_val2 = 0;
		int startj = 0;
		double rate = 10;// 第二行超第一行的倍数
		double minrate = 0.01;

		int times = 0;
		int maxtimes = 5;// 像素值突然增大为区域开始
		boolean isend = false;
		for (int j = 0; j < imgheight; j++) {
			double num = hist.get(0, j)[0];

			if (!isstart) {
				if (num > 0) {
					if (start_val1 == 0)
						start_val1 = num;
					else {
						start_val2 = num;

						// 找到开始
						if (start_val2 / start_val1 >= 2) {
							isstart = true;
							
						} else {
							// 重新开始计算
							start_val1 = num;
						}
					}

				}

				if (isstart) {
					startj = j;
					Rect r = new Rect(0, j, imgwidth, 0);

					rects.add(r);

				}

			} else {

				if (num > 0) {
					if (start_val1 == 0)
						start_val1 = num;
					else {
						start_val2 = num;

						
						//TODO

						// 找到结束
						if (start_val1 / start_val2 < 3    ) {
							isend = true;
							start_val1 = 0;
						} else {
							// 重新开始计算
							start_val1 = num;
						}
						
						
					}

				}

				if (isend) {

					isstart = false;
					isend=false;
					start_val1 = 0;

					Rect r = rects.get(rects.size() - 1);
					r.height = j - startj;
					rects.remove(rects.size() - 1);
					rects.add(r);
				}

				/*
				 * start_val2 = num;
				 * 
				 * 
				 * if (start_val2/max<minrate && start_val2/start_val1>2&& j - startj>10) {
				 * 
				 * isstart=false;
				 * 
				 * start_val1=0;
				 * 
				 * 
				 * Rect r = rects.get(rects.size() - 1); r.height = j - startj;
				 * rects.remove(rects.size() - 1); rects.add(r); } else { start_val1 = num; }
				 */

			}

		}

		return rects;

	}

	/**
	 * 计算xy投影
	 * 
	 * 输入二值化/rect切割好的证件图片
	 * 
	 * @author zj
	 * @date 2018年8月7日
	 */
	private Mat canx(Mat input) {

		int imgheight = input.height();
		int imgwidth = input.width();

		Mat hist = new Mat(1, imgwidth, CvType.CV_32FC1);

		Core.normalize(hist, hist, 0, hist.rows(), Core.NORM_MINMAX, -1, new Mat());
		double[] v = new double[imgwidth];

		for (int j = 0; j < imgwidth; j++) {
			v[j] = 0;
		}
		hist.put(0, 0, v);

		double max = imgheight;// 0;
		for (int j = 0; j < imgwidth; j++) {
			for (int i = 0; i < imgheight; i++) {

				if (input.get(i, j)[0] == 255) {

					double num = hist.get(0, j)[0];
					num++;
					hist.put(0, j, num);
				} else {
					int k = 0;
				}
			}
			double num = hist.get(0, j)[0];
			if (num > max)
				max = num;
		}

		Mat hist2 = new Mat(imgheight, imgwidth, CvType.CV_8UC1);

		Core.normalize(hist2, hist2, 0, hist2.rows(), Core.NORM_MINMAX, -1, new Mat());
		double[] v2 = new double[imgwidth * imgheight];

		for (int j = 0; j < imgwidth * imgheight; j++) {
			v2[j] = 255;
		}
		hist2.put(0, 0, v2);
		for (int j = 0; j < imgwidth; j++) {

			double y = (hist.get(0, j)[0] / max) * imgheight;

			// B component or gray image
			Imgproc.line(hist2, new Point(j, imgheight), new Point(j, y), new Scalar(0, 0, 0), 2, 8, 0);
		}

		// Imgproc.cvtColor(hist2, hist2, Imgproc.COLOR_GRAY2BGR);

		return hist2;

	}

	private void showImg(ImageView view, Mat m) {
		// show the image
		this.updateImageView(view, Utils.mat2Image(m));
		// set a fixed width
		view.setFitWidth(450);

		view.setFitHeight(350);

	}

	/**
	 * 区域识别
	 * 
	 * @author zj
	 * @date 2018年8月6日
	 */
	@FXML
	private void canculRange() {

		Mat m = new Mat();
		Mat filtered = new Mat();
		Mat edges = new Mat();
		Mat dilated_edges = new Mat();
		this.grayimage.copyTo(m);

		// 滤波，模糊处理，消除某些背景干扰信息
		Imgproc.blur(m, filtered, new Size(3, 3));

		savetoImg(filtered, "blur_33");

		// 腐蚀操作，消除某些背景干扰信息
		Imgproc.erode(filtered, filtered, new Mat(), new Point(-1, -1), 3);// 1, 1);

		savetoImg(filtered, "blur_33_erode");

		// 获取直方图均衡化后的灰度像素均值.
		double mean = getthresh(m);

		// double thresh =mean;
		double thresh = this.threshold.getValue();// 25D;

		// this.grayimage.copyTo(filtered);

		// mean=194.49;

		/*
		 * double threshmin=mean*0.44; double threshmax=mean*1.33;
		 */

		double threshmin = thresh;
		double threshmax = threshmin * 3;

		System.out.println("mean:" + mean + "/threshmin:" + threshmin + "/threshmax:" + threshmax);

		// 边缘检测
		Imgproc.Canny(filtered, edges, threshmin, threshmax);

		savetoImg(edges, "blur_33_erode_edges");

		// show the image
		this.updateImageView(antitransformedImage, Utils.mat2Image(edges));
		// set a fixed width
		this.antitransformedImage.setFitWidth(450);

		// 膨胀操作，尽量使边缘闭合
		Imgproc.dilate(edges, dilated_edges, new Mat(), new Point(-1, -1), 3);// , 1, 1);

		savetoImg(dilated_edges, "blur_33_erode_edges_dilated_edges");

	}

	/**
	 * 绘制边框矩形，
	 * 
	 * @param dilated_edges
	 * @author zj
	 * @date 2018年8月7日
	 */
	private Rect dorect(Mat dilated_edges, Mat o) {

		// Mat o = new Mat();
		// this.image.copyTo(o);

		List<MatOfInt> hulls = new ArrayList<>();
		List<MatOfPoint> squares = new ArrayList<>();

		MatOfInt hull = new MatOfInt();
		MatOfPoint approx = new MatOfPoint();

		// find contours

		// dilated_edges 二值化处理后的图像，比如candy/threshold等
		List<MatOfPoint> contours = new ArrayList<>();// 每一组Point点集就是一个轮廓。
		// 向量内每个元素保存了一个包含4个int整型的数,hierarchy向量内每一个元素的4个int型变量——hierarchy[i][0]
		// ~hierarchy[i][3]，分别表示第
		// i个轮廓的后一个轮廓、前一个轮廓、父轮廓、内嵌轮廓的索引编号,没有值则为-1
		Mat hierarchy = new Mat();
		// RETR_EXTERNAL:表示只检测最外层轮廓，对所有轮廓设置hierarchy[i][2]=hierarchy[i][3]=-1
		// RETR_LIST:提取所有轮廓，并放置在list中，检测的轮廓不建立等级关系
		// RETR_CCOMP:提取所有轮廓，并将轮廓组织成双层结构(two-level hierarchy),顶层为连通域的外围边界，次层位内层边界
		// RETR_TREE:提取所有轮廓并重新建立网状轮廓结构
		// RETR_FLOODFILL：官网没有介绍，应该是洪水填充法

		// CHAIN_APPROX_NONE：获取每个轮廓的每个像素，相邻的两个点的像素位置差不超过1
		// CHAIN_APPROX_SIMPLE：压缩水平方向，垂直方向，对角线方向的元素，值保留该方向的重点坐标，如果一个矩形轮廓只需4个点来保存轮廓信息
		// CHAIN_APPROX_TC89_L1和CHAIN_APPROX_TC89_KCOS使用Teh-Chinl链逼近算法中的一种
		// 寻找边界轮廓
		Imgproc.findContours(dilated_edges, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

		// For conversion later on
		MatOfPoint2f approxCurve = new MatOfPoint2f();

		Rect rect = new Rect();

		// For each contour found
		for (int i = 0; i < contours.size(); i++) {

			// Convert contours from MatOfPoint to MatOfPoint2f
			MatOfPoint2f contour2f = new MatOfPoint2f(contours.get(i).toArray());

			// Imgproc.drawMarker(o, contour2f, i, new Scalar(255, 0, 0, .8), 6);

			// Imgproc.drawContours(o, contours, i, new Scalar(255, 255, 0, .8), 6);

			// Processing on mMOP2f1 which is in type MatOfPoint2f
			// 计算轮廓的长度
			double approxDistance = Imgproc.arcLength(contour2f, true) * 0.02;

			if (approxDistance > 1) {
				// Find Polygons
				// 连续轮廓折线化
				Imgproc.approxPolyDP(contour2f, approxCurve, approxDistance, true);

				// Convert back to MatOfPoint
				MatOfPoint points = new MatOfPoint(approxCurve.toArray());

				// Rectangle Checks - Points, area, convexity
				// 矩形检测，4个顶点，凸边形，有一定的面积.
				if (points.total() == 4 && Math.abs(Imgproc.contourArea(points)) > 1000
						&& Imgproc.isContourConvex(points)) {
					double cos = 0;
					double mcos = 0;
					for (int sc = 2; sc < 5; sc++) {
						// TO-DO Figure a way to check angle
						if (cos > mcos) {
							mcos = cos;
						}
					}
					if (mcos < 0.3) {
						// Get bounding rect of contour
						rect = Imgproc.boundingRect(points);

						// if (Math.abs(rect.height - rect.width) < 1000) {
						System.out.println(i + "| x: " + rect.x + " + width(" + rect.width + "), y: " + rect.y
								+ "+ width(" + rect.height + ")");
						// rects.add(rect);
						// Imgproc.rectangle(m, rect.tl(), rect.br(), new Scalar(20, 20, 20), -1, 4, 0);

						// 绘制边界轮廓
						Imgproc.drawContours(o, contours, i, new Scalar(255, 0, 0, .8), 6);

						// Highgui.imwrite("detected_layers"+i+".png", originalImage);
						// }
					}
				}
			}
		}

		// savetoImg(dilated_edges, "contour");

		return rect;

	}

	public boolean isContourSquare(MatOfPoint thisContour) {

		Rect ret = null;

		MatOfPoint2f thisContour2f = new MatOfPoint2f();
		MatOfPoint approxContour = new MatOfPoint();
		MatOfPoint2f approxContour2f = new MatOfPoint2f();

		thisContour.convertTo(thisContour2f, CvType.CV_32FC2);

		Imgproc.approxPolyDP(thisContour2f, approxContour2f, 2, true);

		approxContour2f.convertTo(approxContour, CvType.CV_32S);

		if (approxContour.size().height == 4) {
			ret = Imgproc.boundingRect(approxContour);
		}

		return (ret != null);
	}

	public List<MatOfPoint> getSquareContours(List<MatOfPoint> contours) {

		List<MatOfPoint> squares = null;

		for (MatOfPoint c : contours) {

			if (isContourSquare(c)) {

				if (squares == null)
					squares = new ArrayList<MatOfPoint>();
				squares.add(c);
			}
		}

		return squares;
	}

	/**
	 * Apply Canny
	 * 
	 * @param frame
	 *            the current frame
	 * @return an image elaborated with Canny
	 */
	private Mat doCanny(Mat frame) {
		// init
		Mat grayImage = new Mat();
		Mat detectedEdges = new Mat();

		// convert to grayscale
		Imgproc.cvtColor(frame, grayImage, Imgproc.COLOR_BGR2GRAY);

		// reduce noise with a 3x3 kernel
		Imgproc.blur(grayImage, detectedEdges, new Size(3, 3));

		// canny detector, with ratio of lower:upper feshold of 3:1
		Imgproc.Canny(detectedEdges, detectedEdges, this.threshold.getValue(), this.threshold.getValue() * 3);

		String valuesToPrint = "threshValue: " + this.threshold.getValue();
		Utils.onFXThread(this.hsvValuesProp, valuesToPrint);

		// using Canny's output as a mask, display the result
		Mat dest = new Mat();
		frame.copyTo(dest, detectedEdges);

		return dest;
	}

	/**
	 * Load an image from disk
	 */
	@FXML
	protected void loadImage() {
		// show the open dialog window
		File file = this.fileChooser.showOpenDialog(this.stage);
		if (file != null) {
			// read the image in gray scale
			// this.image = Imgcodecs.imread(file.getAbsolutePath(),
			// Imgcodecs.CV_LOAD_IMAGE_GRAYSCALE);

			// this.image = Imgcodecs.imread(file.getAbsolutePath(),
			// Imgcodecs.IMREAD_COLOR);
			this.image = Imgcodecs.imread(file.getAbsolutePath());// , Imgcodecs.IMREAD_COLOR);
			this.grayimage = Imgcodecs.imread(file.getAbsolutePath(), Imgcodecs.CV_LOAD_IMAGE_GRAYSCALE);

			// System.out.println("normal img:"+this.image.get(0, 0).length);
			// System.out.println("gray img:"+ this.grayimage.get(0, 0).length);

			// show the image
			this.updateImageView(originalImage, Utils.mat2Image(this.image));
			// set a fixed width
			this.originalImage.setFitWidth(450);
			this.originalImage.setFitHeight(350);
			/*
			 * // preserve image ratio this.originalImage.setPreserveRatio(true); // update
			 * the UI this.transformButton.setDisable(false);
			 * 
			 * // empty the image planes and the image views if it is not the first //
			 * loaded image if (!this.planes.isEmpty()) { this.planes.clear();
			 * this.transformedImage.setImage(null);
			 * this.antitransformedImage.setImage(null); }
			 */

		}
	}

	@FXML
	private void hebing() {

		int imgheight = this.image.height();
		int imgwidth = this.image.width();

		Mat src1 = new Mat();
		Mat src2 = new Mat();
		this.image.copyTo(src1);
		this.grayimage.copyTo(src2);

		Mat frame = new Mat(imgwidth * 2, imgheight * 2, this.image.type());

		Rect roiArea = new Rect(0, 0, imgwidth, imgheight);
		Rect roiArea2 = new Rect(0, imgheight, imgwidth, imgheight);
		Rect roiArea3 = new Rect(imgwidth, 0, imgwidth, imgheight);
		Rect roiArea4 = new Rect(imgwidth, imgheight, imgwidth, imgheight);

		Mat src1Roi = new Mat(src1, roiArea);
		Mat dstRoi = new Mat(frame, roiArea);
		Core.add(src1Roi, dstRoi, dstRoi);

		dstRoi = new Mat(frame, roiArea2);
		Core.add(src1, dstRoi, dstRoi);

		dstRoi = new Mat(frame, roiArea3);

		Mat matc = new Mat();
		Imgproc.cvtColor(src2, matc, Imgproc.COLOR_GRAY2BGR);
		Core.add(matc, dstRoi, dstRoi);

		dstRoi = new Mat(frame, roiArea4);
		Core.add(src1, dstRoi, dstRoi);

		double alpha = 0.5;
		double beta;
		double input;

		beta = (1.0 - alpha);
		// Core.addWeighted( src1, alpha, src2, beta, 0.0, frame);

		// Core.add
		// Core.add(src1Roi, src1Roi, dstRoi, dstRoi, this.orimage.type());

		// show the result of the transformation as an image
		this.updateImageView(transformedImage, Utils.mat2Image(frame));
		// set a fixed width
		this.transformedImage.setFitWidth(650);
		this.transformedImage.setFitHeight(600);

	}

	@FXML
	private void faceDetect() {
		Mat src1 = new Mat();

		this.image.copyTo(src1);

		dodetectAndDisplay(src1);
	}

	private void savetoImg(Mat m, String name) {

		String path = FourierController.class.getResource("/").getPath();
		File f = new File(path + "\\out");
		if (!f.exists())
			f.mkdirs();

		Imgcodecs.imwrite(f.getAbsolutePath() + "\\" + name + ".png", m);

	}

	private void dodetectAndDisplay(Mat input) {

		Mat frame = new Mat();
		input.copyTo(frame);

		MatOfRect faces = new MatOfRect();
		Mat grayFrame = new Mat();

		// init
		// Mat grayFrame = this.image;

		// convert the frame in gray scale
		Imgproc.cvtColor(frame, grayFrame, Imgproc.COLOR_BGR2GRAY);
		// equalize the frame histogram to improve the result
		// //函数功能：直方图均衡化，该函数能归一化图像亮度和增强对比度
		Imgproc.equalizeHist(grayFrame, grayFrame);

		// show the result of the transformation as an image
		this.updateImageView(antitransformedImage, Utils.mat2Image(grayFrame));
		// set a fixed width
		this.antitransformedImage.setFitWidth(450);

		// compute minimum face size (20% of the frame height, in our case)
		if (this.absoluteFaceSize == 0) {
			int height = grayFrame.rows();
			if (Math.round(height * 0.2f) > 0) {
				this.absoluteFaceSize = Math.round(height * 0.2f);
			}
		}

		// detect faces
		this.faceCascade.detectMultiScale(grayFrame, faces, 1.1, 2, 0 | Objdetect.CASCADE_SCALE_IMAGE,
				new Size(this.absoluteFaceSize, this.absoluteFaceSize), new Size());

		// each rectangle in faces is a face: draw them!
		Rect[] facesArray = faces.toArray();
		for (int i = 0; i < facesArray.length; i++)
			Imgproc.rectangle(frame, facesArray[i].tl(), facesArray[i].br(), new Scalar(0, 255, 0), 3);

		// show the result of the transformation as an image
		this.updateImageView(transformedImage, Utils.mat2Image(frame));
		// set a fixed width
		this.transformedImage.setFitWidth(450);

	}

	/**
	 * The action triggered by pushing the button for apply the dft to the loaded
	 * image
	 */
	@FXML
	protected void transformImage() {
		// optimize the dimension of the loaded image
		Mat padded = this.optimizeImageDim(this.image);
		padded.convertTo(padded, CvType.CV_32F);
		// prepare the image planes to obtain the complex image
		this.planes.add(padded);
		this.planes.add(Mat.zeros(padded.size(), CvType.CV_32F));
		// prepare a complex image for performing the dft
		Core.merge(this.planes, this.complexImage);

		// dft
		Core.dft(this.complexImage, this.complexImage);

		// optimize the image resulting from the dft operation
		Mat magnitude = this.createOptimizedMagnitude(this.complexImage);

		// show the result of the transformation as an image
		this.updateImageView(transformedImage, Utils.mat2Image(magnitude));
		// set a fixed width
		this.transformedImage.setFitWidth(250);
		// preserve image ratio
		this.transformedImage.setPreserveRatio(true);

		// enable the button for performing the antitransformation
		this.antitransformButton.setDisable(false);
		// disable the button for applying the dft
		this.transformButton.setDisable(true);
	}

	/**
	 * The action triggered by pushing the button for apply the inverse dft to the
	 * loaded image
	 */
	@FXML
	protected void antitransformImage() {
		Core.idft(this.complexImage, this.complexImage);

		Mat restoredImage = new Mat();
		Core.split(this.complexImage, this.planes);
		Core.normalize(this.planes.get(0), restoredImage, 0, 255, Core.NORM_MINMAX);

		// move back the Mat to 8 bit, in order to proper show the result
		restoredImage.convertTo(restoredImage, CvType.CV_8U);

		this.updateImageView(antitransformedImage, Utils.mat2Image(restoredImage));
		// set a fixed width
		this.antitransformedImage.setFitWidth(250);
		// preserve image ratio
		this.antitransformedImage.setPreserveRatio(true);

		// disable the button for performing the antitransformation
		this.antitransformButton.setDisable(true);
	}

	/**
	 * Optimize the image dimensions
	 * 
	 * @param image
	 *            the {@link Mat} to optimize
	 * @return the image whose dimensions have been optimized
	 */
	private Mat optimizeImageDim(Mat image) {
		// init
		Mat padded = new Mat();
		// get the optimal rows size for dft
		int addPixelRows = Core.getOptimalDFTSize(image.rows());
		// get the optimal cols size for dft
		int addPixelCols = Core.getOptimalDFTSize(image.cols());
		// apply the optimal cols and rows size to the image
		Core.copyMakeBorder(image, padded, 0, addPixelRows - image.rows(), 0, addPixelCols - image.cols(),
				Core.BORDER_CONSTANT, Scalar.all(0));

		return padded;
	}

	/**
	 * Optimize the magnitude of the complex image obtained from the DFT, to improve
	 * its visualization
	 * 
	 * @param complexImage
	 *            the complex image obtained from the DFT
	 * @return the optimized image
	 */
	private Mat createOptimizedMagnitude(Mat complexImage) {
		// init
		List<Mat> newPlanes = new ArrayList<>();
		Mat mag = new Mat();
		// split the comples image in two planes
		Core.split(complexImage, newPlanes);
		// compute the magnitude
		Core.magnitude(newPlanes.get(0), newPlanes.get(1), mag);

		// move to a logarithmic scale
		Core.add(Mat.ones(mag.size(), CvType.CV_32F), mag, mag);
		Core.log(mag, mag);
		// optionally reorder the 4 quadrants of the magnitude image
		this.shiftDFT(mag);
		// normalize the magnitude image for the visualization since both JavaFX
		// and OpenCV need images with value between 0 and 255
		// convert back to CV_8UC1
		mag.convertTo(mag, CvType.CV_8UC1);
		Core.normalize(mag, mag, 0, 255, Core.NORM_MINMAX, CvType.CV_8UC1);

		// you can also write on disk the resulting image...
		// Imgcodecs.imwrite("../magnitude.png", mag);

		return mag;
	}

	/**
	 * Reorder the 4 quadrants of the image representing the magnitude, after the
	 * DFT
	 * 
	 * @param image
	 *            the {@link Mat} object whose quadrants are to reorder
	 */
	private void shiftDFT(Mat image) {
		image = image.submat(new Rect(0, 0, image.cols() & -2, image.rows() & -2));
		int cx = image.cols() / 2;
		int cy = image.rows() / 2;

		Mat q0 = new Mat(image, new Rect(0, 0, cx, cy));
		Mat q1 = new Mat(image, new Rect(cx, 0, cx, cy));
		Mat q2 = new Mat(image, new Rect(0, cy, cx, cy));
		Mat q3 = new Mat(image, new Rect(cx, cy, cx, cy));

		Mat tmp = new Mat();
		q0.copyTo(tmp);
		q3.copyTo(q0);
		tmp.copyTo(q3);

		q1.copyTo(tmp);
		q2.copyTo(q1);
		tmp.copyTo(q2);
	}

	/**
	 * Set the current stage (needed for the FileChooser modal window)
	 * 
	 * @param stage
	 *            the stage
	 */
	public void setStage(Stage stage) {
		this.stage = stage;
	}

	/**
	 * Update the {@link ImageView} in the JavaFX main thread
	 * 
	 * @param view
	 *            the {@link ImageView} to update
	 * @param image
	 *            the {@link Image} to show
	 */
	private void updateImageView(ImageView view, Image image) {
		Utils.onFXThread(view.imageProperty(), image);
	}

}
