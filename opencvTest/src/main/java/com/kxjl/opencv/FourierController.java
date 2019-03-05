package com.kxjl.opencv;

import java.awt.image.BufferedImage;
import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
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
import org.opencv.imgproc.CLAHE;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.objdetect.Objdetect;
import org.opencv.utils.Converters;
import org.opencv.video.Video;
import org.opencv.videoio.VideoCapture;

import com.sun.scenario.effect.light.Light;

import javafx.beans.property.ObjectProperty;
import javafx.beans.property.SimpleObjectProperty;
import javafx.fxml.FXML;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.Slider;
import javafx.scene.control.TextArea;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.stage.FileChooser;
import javafx.stage.Stage;
import net.sourceforge.tess4j.ITesseract;
import net.sourceforge.tess4j.Tesseract;
import net.sourceforge.tess4j.TesseractException;

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
	private ImageView originalImage2;

	@FXML
	private ImageView antitransformedImage3;
	@FXML
	private ImageView originalImage3;
	@FXML
	private ImageView transformedImage;
	@FXML
	private ImageView antitransformedImage;

	@FXML
	private ImageView transformedImage2;

	@FXML
	private ImageView transformedImage3;
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

	@FXML
	private Slider valueDuibi;
	@FXML
	private Slider valuelight;

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

	@FXML
	private TextArea txtinfo;

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
		Imgproc.pyrUp(frame, frame);
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

		Mat normal = new Mat();
		Imgproc.cvtColor(frame, normal, Imgproc.COLOR_GRAY2BGR);
		double threshValue2 = getAvGray(normal);
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
		threshValue = 95;// (int) this.threshold.getValue(); // 140;//
		// this.getHistAverage(hsvImg, hsvPlanes.get(0));

		System.out.println("threshValue:" + threshValue);
		double to = 255;// threshold2.getValue(); //10;//
		System.out.println("to:" + to);

		int s3 = 2;// (int)threshold3.getValue();
		int s4 = 2;// (int)threshold4.getValue();

		int thresh_type = Imgproc.THRESH_BINARY_INV;
		// threshValue
		// 双值化图像
		Imgproc.threshold(frame, thresholdImg, threshValue, to, thresh_type);

		Imgproc.blur(thresholdImg, thresholdImg, new Size(2, 2));

		Mat structElement1 = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(1, 1), new Point(-1, -1));
		// dilate to fill gaps, erode to smooth edges
		Imgproc.dilate(thresholdImg, thresholdImg, structElement1, new Point(-1, -1), s3);
		Imgproc.erode(thresholdImg, thresholdImg, structElement1, new Point(-1, -1), s4);

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
		this.grayimage.copyTo(nmat);

		Mat t = ruihua(nmat);

		savetoImg(t, "ruihua");

		// show the image
		this.updateImageView(antitransformedImage, Utils.mat2Image(t));

		// set a fixed width
		this.antitransformedImage.setFitWidth(450);
	}

	/**
	 * 图像对比度调整
	 * 
	 * @author zj
	 * @date 2018年8月15日
	 */
	@FXML
	private void t2duibidui() {

		Mat n = new Mat();
		this.image.copyTo(n);

		Mat eqMat = new Mat();
		eqMat = duibidu(n);
		// dst.convertTo(dst, dst.type(), valueDuibi.getValue(), valuelight.getValue());
		showImg(transformedImage2, eqMat);

	}

	private Mat duibidu(Mat data) {

		int imgwidth = image.width();
		int imgheight = image.height();

		List<Mat> bgr = new ArrayList<>();
		Core.split(data, bgr);
		Mat bChannel = bgr.get(0).zeros(image.size(), bgr.get(0).type());
		// Core.normalize(bChannel, bChannel, 0, 255, Core.NORM_MINMAX);
		Mat gChannel = bgr.get(1).zeros(image.size(), bgr.get(0).type());
		// Core.normalize(gChannel, gChannel, 0, 255, Core.NORM_MINMAX);
		Mat rChannel = bgr.get(2);
		// Core.normalize(rChannel, rChannel, 0, 255, Core.NORM_MINMAX);
		Mat dehazedImg = new Mat();
		Core.merge(new ArrayList<>(Arrays.asList(bChannel, gChannel, rChannel)), dehazedImg);

		Mat dst = Mat.zeros(image.size(), image.type());

		System.out.println("valueDuibi:" + valueDuibi.getValue());
		System.out.println("valuelight:" + valuelight.getValue());

		for (int i = 0; i < imgwidth; i++) {

			for (int j = 0; j < imgheight; j++) {

				double bdst = bgr.get(0).get(j, i)[0] * valueDuibi.getValue() + valuelight.getValue();
				double gdst = bgr.get(1).get(j, i)[0] * valueDuibi.getValue() + valuelight.getValue();
				double rdst = bgr.get(2).get(j, i)[0] * valueDuibi.getValue() + valuelight.getValue();
				double[] val = new double[] { bdst, gdst, rdst };
				dst.put(j, i, val);
			}
		}

		Core.normalize(dst, dst, 0, 255, Core.NORM_MINMAX);

		// savetoImg(dst, "duibi");

		// showImg(originalImage2, dehazedImg);

		// showImg(transformedImage, dst);

		// Mat eqMat = new Mat();
		// Imgproc.equalizeHist(bgr.get(0), eqMat);

		return dst;
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
			delval = 125;
		else if (avlight > 160 && avlight <= 170)
			delval = 91;
		else if (avlight > 170 && avlight <= 190)
			delval = 141;
		else if (avlight > 190)
			delval = 141;

		return delval;
	}

	@FXML
	private void hsv() {
		Mat m = new Mat();
		this.image.copyTo(m);

		String msg = ocr(m);
		System.out.println(msg);

		String path = FourierController.class.getResource("/").getPath();
		File f1 = new File(path + "\\out\\ori\\");
		File f = new File(path + "\\out\\ids\\");
		if (!f.exists())
			f.mkdirs();

		if (!f1.exists())
			f1.mkdirs();

		/*
		 * Scalar lower1 = new Scalar(0,150,100); Scalar upper1 = new
		 * Scalar(20,255,255); Scalar lower2 = new Scalar(140,100,100); Scalar upper2 =
		 * new Scalar(179,255,255); Core.inRange(m,lower1,upper1,m);
		 * Core.inRange(m,lower2,upper2,m); Core.addWeighted(m,1.0, m,1.0, 0.0, m);
		 * 
		 * showImg(transformedImage3, m);
		 */

		// Imgproc.pyrUp(m, m);
		// Imgproc.pyrUp(m, m);
		Imgproc.pyrUp(m, m);

		Mat structElement1 = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3, 3), new Point(-1, -1));
		/*
		 * Mat structElement2 = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new
		 * Size(1,1), new Point(-1, -1));
		 */
		Imgproc.erode(m, m, structElement1, new Point(-1, -1), 3);// 1, 1);

		Imgproc.dilate(m, m, structElement1, new Point(-1, -1), 3);
		Imgproc.pyrDown(m, m);
		// Imgcodecs.imwrite(f.getAbsolutePath() + "\\" + name + "_b.png", m);
		m = ruihua(m);
		Imgproc.blur(m, m, new Size(3, 3));
		showImg(antitransformedImage2, m);

		// 18-103, 82-161
		m = getSimpleImgForJashiz(m, 0, 0);

		Imgproc.pyrDown(m, m);
		showImg(antitransformedImage3, m);

		savetoImg(m, "test");

		m = ruihua(m);

		// RuihuaAndsavetoImg(in,"test");

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

		/*
		 * Scalar minValues = new Scalar(starth, this.saturationStart.getValue(),
		 * this.valueStart.getValue()); Scalar maxValues = new Scalar(endh,
		 * this.saturationStop.getValue(), this.valueStop.getValue());
		 */

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
	 * 获取灰度图的中间1/3区域宽度的平均灰度值
	 * 
	 * @param input
	 *            灰度图
	 * @return
	 * @author zj
	 * @date 2018年8月7日
	 */
	private double getAvThirdWidthGray(Mat input) {
		double v = 0;

		Mat gray = new Mat();
		// input.copyTo(gray);
		Imgproc.cvtColor(input, gray, Imgproc.COLOR_BGR2GRAY);

		// //函数功能：直方图均衡化，该函数能归一化图像亮度和增强对比度
		// Imgproc.equalizeHist(gray, gray);

		int imgwidth = gray.width();
		int imgheight = gray.height();
		double total = 0;
		for (int i = imgwidth / 3; i < imgwidth * 2 / 3; i++) {
			for (int j = 0; j < imgheight; j++) {
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
	 * 获取输入bgr图片的 平均灰度值
	 * 
	 * @param input
	 *            bgr图
	 * @return
	 * @author zj
	 * @date 2018年8月7日
	 */
	private double getAvGray(Mat input) {
		double v = 0;

		Mat gray = new Mat();
		// input.copyTo(gray);
		Imgproc.cvtColor(input, gray, Imgproc.COLOR_BGR2GRAY);

		// //函数功能：直方图均衡化，该函数能归一化图像亮度和增强对比度
		// Imgproc.equalizeHist(gray, gray);

		int imgwidth = gray.width();
		int imgheight = gray.height();
		double total = 0;
		for (int i = 0; i < imgwidth; i++) {
			for (int j = 0; j < imgheight; j++) {
				// int j=imgheight/2;
				double[] vals = gray.get(j, i);
				double light = vals[0];
				total += light;

			}

		}
		v = total / (imgwidth * imgheight);
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
		Imgproc.cvtColor(input, gray, Imgproc.COLOR_BGR2GRAY);

		// //函数功能：直方图均衡化，该函数能归一化图像亮度和增强对比度
		// Imgproc.equalizeHist(gray, gray);

		int imgwidth = gray.width();
		int imgheight = gray.height();
		double max = 0;
		for (int i = imgwidth / 3; i < imgwidth * 2 / 3; i++) {
			for (int j = 0; j < imgheight; j++) {
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

	/**
	 * bgr 适合尺寸 原始图片 身份证识别
	 * 
	 * @param input
	 * @return
	 * @author zj
	 * @date 2018年8月9日
	 */
	private Mat twoValueIdCard(Mat input) {
		Mat rectM = new Mat();

		Mat m = new Mat();
		Mat filtered = new Mat();
		// this.image.copyTo(rectM);
		input.copyTo(rectM);

		List<Mat> planes = new ArrayList<>();
		Core.split(rectM, planes);

		showImg(antitransformedImage, planes.get(0));
		showImg(transformedImage, planes.get(1));
		showImg(transformedImage2, planes.get(2));

		/*
		 * List<Mat> planes=new ArrayList<>(); Core.split(rectM, planes);
		 * planes.get(2).copyTo(m);
		 * 
		 * 
		 * m.copyTo(filtered); // 滤波，模糊处理，消除某些背景干扰信息
		 * 
		 * Imgproc.blur(m, filtered, new Size(3, 3));
		 * 
		 * savetoImg(rectM, "1");
		 * 
		 * Mat structElement1 = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new
		 * Size(6, 6)); // 腐蚀操作，消除某些背景干扰信息 Imgproc.erode(filtered, filtered,
		 * structElement1, new Point(-1, -1), 1);// 1, 1); savetoImg(filtered, "2");
		 * Imgproc.dilate(filtered, filtered, structElement1, new Point(-1, -1), 1);
		 * savetoImg(filtered, "3");
		 */

		// Mat hist2= blacTwovalue(rectM);

		Mat hist2 = blacTwovalue2(planes.get(0), 0);

		// savetoImg(hist2, "4");

		showImg(antitransformedImage3, hist2);

		return hist2;
	}

	@FXML
	private void dotwovalue() {

		Mat m = new Mat();
		this.grayimage.copyTo(m);

		Mat newm = blacTwovalue2(m, 90);
		System.out.println(ocr(newm));
		Imgproc.pyrUp(newm, newm);

		newm = ruihua(newm);

		this.updateImageView(transformedImage, Utils.mat2Image(newm));
		// set a fixed width
		transformedImage.setFitWidth(newm.width());
		transformedImage.setFitHeight(newm.height());

		showImg(antitransformedImage, newm);

		System.out.println(ocr(newm));

	}

	/**
	 * 计算灰度图像像素黑度 灰度计算二值化方法
	 * 
	 * @param rectM
	 * @return
	 * @author zj
	 * @date 2018年8月16日
	 */
	private Mat blacTwovalue2(Mat rectM, int threldval) {
		int imgheight = rectM.height();
		int imgwidth = rectM.width();

		Mat hist2 = new Mat(imgheight, imgwidth, CvType.CV_8UC1);

		Core.normalize(hist2, hist2, 0, hist2.rows(), Core.NORM_MINMAX, -1, new Mat());
		double[] v2 = new double[imgwidth * imgheight];

		for (int j = 0; j < imgwidth * imgheight; j++) {
			v2[j] = 255;
		}
		hist2.put(0, 0, v2);

		Mat bgr = new Mat();
		Imgproc.cvtColor(rectM, bgr, Imgproc.COLOR_GRAY2BGR);

		// 获取灰度图的中间1/3区域宽度的平均灰度值
		double avlihghtThird = getAvThirdWidthGray(bgr);
		// 获取输入bgr图片的 平均灰度值
		double avlihght = getAvGray(bgr);
		// 获取灰度图的1/3区域高度的最大灰度值
		double maxlihght = getMaxthresh(bgr);

		double val = maxlihght - avlihghtThird;// 差值
		System.out.println("avlihghtThird:" + avlihghtThird);
		System.out.println("avlihght:" + avlihght);
		System.out.println("maxlihght:" + maxlihght);
		System.out.println(" maxlihght - avlihghtThird:" + val);
		System.out.println(" Math.abs(avlihght - avlihghtThird):" + Math.abs(avlihght - avlihghtThird));

		/*
		 * double param=5; if(avlihght>175) param= avlihght-130;
		 * 
		 */
		double param = (255 - maxlihght) * 2;

		// 确定过滤的灰色阈值
		if (Math.abs(avlihght - avlihghtThird) < 20) {

			// 中间区域与全图平均灰度差别不大，无高亮/图片颜色分布均衡

			if (val < 35) // 对比度比较低，整体偏亮
			{
				if (avlihghtThird < 150) // 平均灰度第，颜色清晰
					param = 115; // 过滤偏黑灰色.
				else if (avlihghtThird >= 150 && avlihghtThird < 180) // 平均灰度第，颜色清晰
					param = 135; // 过滤偏黑灰色.
				else
					param = 149; // //平均灰度第，颜色不太清晰

			} else if (val >= 35 && val < 61) // 正常图片，颜色分布均衡
			{
				param = 110; // 过滤偏黑灰色.

				if (maxlihght < 200)
					param = 85; // 过滤偏黑灰色.

			} else if (val >= 61 && val < 91) // 对比度比较高，色彩区分度高
				param = 60; // 过滤偏黑色.
			else if (val >= 91) // 对比度超高，图片有亮度不均衡或者部分区域高亮
			{
				param = 60; // 过滤偏黑色.
			}
		}
		System.out.println("param:" + param + " / val:" + threldval);
		if (threldval != 0)
			param = threldval;

		double max = imgheight;// 0;
		for (int j = 0; j < imgwidth; j++) {
			for (int i = 0; i < imgheight; i++) {

				try {
					// if (rectM.get(i, j)[0] <= 100+param && rectM.get(i, j)[1] <= 100+param &&
					// rectM.get(i, j)[2] <= 100+param) {

					// 对彩色图片过滤黑色阈值
					if (rectM.get(i, j)[0] <= param) {

						hist2.put(i, j, 0);
					} else {
						// int k = 0;
					}
				} catch (Exception e) {
					// TODO: handle exception
				}

			}
		}

		return hist2;
	}

	/**
	 * 灰度计算二值化方法
	 * 
	 * @param rectM
	 * @return
	 * @author zj
	 * @date 2018年8月16日
	 */
	private Mat blacTwovalue(Mat rectM) {
		int imgheight = rectM.height();
		int imgwidth = rectM.width();

		Mat hist2 = new Mat(imgheight, imgwidth, CvType.CV_8UC1);

		Core.normalize(hist2, hist2, 0, hist2.rows(), Core.NORM_MINMAX, -1, new Mat());
		double[] v2 = new double[imgwidth * imgheight];

		for (int j = 0; j < imgwidth * imgheight; j++) {
			v2[j] = 255;
		}
		hist2.put(0, 0, v2);

		double avlihghtThird = getAvThirdWidthGray(rectM);
		double avlihght = getAvGray(rectM);
		double maxlihght = getMaxthresh(rectM);

		double val = maxlihght - avlihghtThird;// 差值
		System.out.println("avlight:" + avlihght + " maxgray:" + maxlihght + "  val:" + val);

		/*
		 * double param=5; if(avlihght>175) param= avlihght-130;
		 * 
		 */

		double param = (255 - maxlihght) * 2;

		// 确定过滤的灰色阈值
		if (Math.abs(avlihght - avlihghtThird) < 20) {

			// 中间区域与全图平均灰度差别不大，无高亮/图片颜色分布均衡

			if (val < 35) // 对比度比较低，整体偏亮
			{
				if (avlihghtThird < 150) // 平均灰度第，颜色清晰
					param = 125; // 过滤偏黑灰色.
				if (avlihghtThird >= 150 && avlihghtThird < 180) // 平均灰度第，颜色清晰
					param = 135; // 过滤偏黑灰色.
				else
					param = 149; // //平均灰度第，颜色不太清晰

			} else if (val >= 35 && val < 61) // 正常图片，颜色分布均衡
			{
				param = 110; // 过滤偏黑灰色.

				if (maxlihght < 200)
					param = 85; // 过滤偏黑灰色.

			} else if (val >= 61 && val < 91) // 对比度比较高，色彩区分度高
				param = 60; // 过滤偏黑色.
			else if (val >= 91) // 对比度超高，图片有亮度不均衡或者部分区域高亮
			{
				param = 60; // 过滤偏黑色.
			}
		}

		double max = imgheight;// 0;
		for (int j = 0; j < imgwidth; j++) {
			for (int i = 0; i < imgheight; i++) {

				try {
					// if (rectM.get(i, j)[0] <= 100+param && rectM.get(i, j)[1] <= 100+param &&
					// rectM.get(i, j)[2] <= 100+param) {

					// 对彩色图片过滤黑色阈值
					if (rectM.get(i, j)[0] <= param && rectM.get(i, j)[1] <= param && rectM.get(i, j)[2] <= param) {

						hist2.put(i, j, 0);
					} else {
						// int k = 0;
					}
				} catch (Exception e) {
					// TODO: handle exception
				}

			}
		}

		return hist2;
	}

	/**
	 * 直接原始全图 身份证 提取黑色字段识别
	 * 
	 * @author zj
	 * @date 2018年8月7日
	 */
	@FXML
	private void caculBack() {

		Mat m = new Mat();
		Mat filtered = new Mat();
		Mat thresholdImg = new Mat();
		Mat dilated_edges = new Mat();

		this.image.copyTo(m);

		doCardReg(m);
	}

	/**
	 * 对输入身份证身份证识别
	 * 
	 * @param input
	 * @author zj
	 * @date 2018年8月9日
	 */
	private void doCardReg(Mat input) {
		Mat m = new Mat();
		input.copyTo(m);

		Mat hist2 = twoValueIdCard(m);

		/*
		 * if (true) return;
		 */

		// 滤波，模糊处理，消除某些背景干扰信息
		Imgproc.blur(hist2, hist2, new Size(1, 1));

		// 腐蚀操作，消除某些背景干扰信息
		Imgproc.erode(hist2, hist2, new Mat(), new Point(-1, -1), 1);// 1, 1);
		Imgproc.dilate(hist2, hist2, new Mat(), new Point(-1, -1), 1);

		cacul2ValMat(m, hist2);
	}

	/**
	 * 锁IMEI
	 * 
	 * @author zj
	 * @date 2019年3月4日
	 */
	@FXML
	private void lockImeiOcr() {

		Mat m = new Mat();
		Mat filtered = new Mat();
		Mat thresholdImg = new Mat();
		
		Mat filtered2 = new Mat();
		Mat thresholdImg2 = new Mat();
		Mat dilated_edges = new Mat();
		
		Mat cutMat=new Mat();

		this.grayimage.copyTo(m);

		m.copyTo(filtered);
		// 滤波，模糊处理，消除某些背景干扰信息
		Imgproc.blur(m, filtered, new Size(1, 1));

		// 腐蚀操作，消除某些背景干扰信息
		// Imgproc.erode(filtered, filtered, new Mat(), new Point(-1, -1), 1);// 1, 1);

		// double maxgrayval = getMaxthresh(m);
		double maxgrayval = 40;// getthresh(m);
		System.out.println("maxgrayval:" + maxgrayval);

		// Imgproc.equalizeHist(filtered, filtered);

		// int thresh_type = Imgproc.THRESH_BINARY_INV; //反转
		int thresh_type = Imgproc.THRESH_OTSU;// 前后背景区分
		// threshValue
		// 双值化图像
		double to = 255;
		Imgproc.threshold(filtered, thresholdImg, maxgrayval * 3 / 4, to, thresh_type);

		// double f=200;
		// to=220;
		// Imgproc.threshold(filtered, thresholdImg, f, to, thresh_type);

		cutMat=preCamcuIMEI(thresholdImg);

		savetoImg(cutMat, "lock_thresholdImg");
		
		//cutMat
		Imgproc.cvtColor(cutMat, filtered2, Imgproc.COLOR_BGR2GRAY);
		//对有裁剪的重新计算二值化，剔除高光等
		 
		Imgproc.threshold(filtered2, thresholdImg2, maxgrayval * 3 / 4, to, thresh_type);
		
		savetoImg(thresholdImg2, "lock_thresholdImg2");
		
		Mat rectM_tp = new Mat();
		cutMat.copyTo(rectM_tp);
		Mat transMat_tp = new Mat();
		Rect r_tp = doContours(thresholdImg2, rectM_tp, transMat_tp);
		
		savetoImg(rectM_tp, "rectM_tp");
		
		 //图像横竖判断。
		 if(	r_tp.width!= 0 && r_tp.height != 0&&r_tp.width<r_tp.height)
		 {
			 //竖着拍的，
			 //翻转图像

				Point center =new Point(cutMat.width()/2.0,cutMat.height()/2.0);
				Mat affineTrans=Imgproc.getRotationMatrix2D(center, 85.0, 1.0);
				
				Imgproc.warpAffine(cutMat, cutMat, affineTrans, cutMat.size(),Imgproc.INTER_NEAREST);
				savetoImg(cutMat, "lock_thresholdImg2_rate");
				
				//cutMat
				Imgproc.cvtColor(cutMat, filtered2, Imgproc.COLOR_BGR2GRAY);
				//对有裁剪的重新计算二值化，剔除高光等
				 
				Imgproc.threshold(filtered2, thresholdImg2, maxgrayval * 3 / 4, to, thresh_type);
				
				
		 }
		
		
		
		/*
		 * if(true) return;
		 */

		Mat rectM = new Mat();
		cutMat.copyTo(rectM);
		Mat rectM2 = new Mat();
		cutMat.copyTo(rectM2);

		// 变换参数
		Mat transMat = new Mat();

		// 截取证件区域数据并进行透视变换为 垂直视角
		Mat realRect = new Mat();
		// Imgproc. warpPerspective(new Mat(rectM2, r),realRect,transMat,new
		// Size(r.width,r.height), Imgproc.INTER_LINEAR + Imgproc.WARP_INVERSE_MAP);

		Rect r = doContours(thresholdImg2, rectM, transMat);
		
		 //图像横竖判断。
		 if(r.width<r.height)
		 {
			 //竖着拍的，
			 //翻转图像

				Point center =new Point(thresholdImg2.width()/2.0,thresholdImg2.height()/2.0);
				Mat affineTrans=Imgproc.getRotationMatrix2D(center, 90.0, 1.0);
				
				Imgproc.warpAffine(thresholdImg2, thresholdImg2, affineTrans, thresholdImg2.size(),Imgproc.INTER_NEAREST);
				savetoImg(thresholdImg2, "lock_thresholdImg2_rate");
				 r = doContours(thresholdImg2, rectM, transMat);
		 }
		
		
		if (r.width == 0 && r.height == 0) {
			// 轮廓计算失败
			r.width = filtered.width();

			r.height = filtered.height();

			rectM2.copyTo(realRect);
		} else {
			// 轮廓ok,变换原图

			// Imgproc.warpPerspective(rectM2, realRect, transMat, rectM2.size(),
			// Imgproc.INTER_LINEAR); // +
			// Imgproc.WARP_INVERSE_MAP
			//

			// 轮廓ok,变换原图
			Imgproc.warpPerspective(rectM2, realRect, transMat, rectM2.size(),
					Imgproc.INTER_LINEAR + Imgproc.WARP_INVERSE_MAP);
		}

		Imgproc.cvtColor(dilated_edges, dilated_edges, Imgproc.COLOR_GRAY2BGR);
		Mat stepM = new Mat();
		dilated_edges.copyTo(stepM);

		Imgproc.rectangle(rectM, r.tl(), r.br(), new Scalar(0, 0, 255), 2);

		showImg(originalImage2, rectM);

		Mat idcarmat = new Mat(realRect, r);

		showImg(originalImage3, idcarmat);

		// 放大
		if (idcarmat.width() < 500)
			Imgproc.pyrUp(idcarmat, idcarmat);

		// doCardReg(idcarmat);

		ocrSimple(idcarmat);

	}

	/**
	 * 通过投影计算<br>
	 * 提出图片中包含imei的一大块方形白色区域<br>
	 * 剔除高光/拍摄的边角等
	 * 
	 * @param input
	 *            输入为二值化后的预处理图片
	 * @return
	 * @author zj
	 * @date 2019年3月5日
	 */
	private Mat preCamcuIMEI(Mat input) {

		Mat m = new Mat();
		Mat idcarmat = new Mat();
		this.image.copyTo(m);

		Mat yhist = new Mat();
		List<Rect> rects = canyForImei(input, yhist);

		// Mat xhist = new Mat();
		// List<Rect> rectxs = canx(input, xhist);

		showImg(antitransformedImage, yhist);
		
		
		if(rects.size()>0)
		{
			//上下增加一下剪切高度 100px，左右缩短宽度
			int pluswidth=100;
			Rect r=new Rect(rects.get(0).x+pluswidth,rects.get(0).y-pluswidth,rects.get(0).width-2*pluswidth,rects.get(0).height+2*pluswidth);
			
		 idcarmat = new Mat(m,  r);
		 
		
		 
		 
		}
		else
			idcarmat=m;
		
		

		// showImg(transformedImage2, xhist);

		return idcarmat;
	}

	/**
	 * 计算xy投影 针对锁的IMEI特定图形 ，剔除高光等影响
	 * 
	 * 输入二值化/rect切割好的证件图片
	 * 
	 * @author zj
	 * @date 2018年8月7日
	 */
	private List<Rect> canyForImei(Mat input, Mat output) {

		int imgheight = input.height();
		int imgwidth = input.width();

		Double max = 0D;
		Mat hist = canyStepOne(input, output, max);

		for (int j = 0; j < imgheight; j++) {
			double num = hist.get(0, j)[0];
			if (max < num)
				max = num;
		}

		// 计算纵向区域
		List<Rect> rects = new ArrayList<>();
		boolean isstart = false;

		double start_val1 = 0;// 前一个y值
		double start_val2 = 0;// 后一个y值
		int startj = 0; // 确定一行开始的y值
		double startv=0;//开始行的值
		
		double rate = 10;// 第二行超第一行的倍数
		double minrate = 0.01;

		int times = 0;
		int maxtimes = 5;// 像素值突然增大为区域开始
		boolean isend = false;
		for (int j = 0; j < imgheight; j++) {
			double num = hist.get(0, j)[0];

			if (!isstart) {
				// 假设开始

				if (start_val1 == 0)
					start_val1 = num;
				else {

					start_val1 = start_val2;
					start_val2 = num;

					// 找到开始
					if (j > 100 &&j<imgheight-100 && start_val1 / start_val2 > 1 && start_val2 < max * 2 / 3) {

						isstart = true;
						startj = j;
						startv=num;
						Rect r = new Rect(0, j, imgwidth, 0);

						rects.add(r);
					} else {
						// 重新开始计算 第一个y
						// start_val1 = num;
					}
				}

			} else {

				if (start_val1 == 0)
					start_val1 = num;
				else {
					start_val1 = start_val2;
					start_val2 = num;

					if (Math.abs(startv - start_val2) < 35)
						times++;// 前后相差不多，计数
					else {

						//差距开始变大
						
						if (times > 100&& start_val2>start_val1) // 连续40个相差不大 ,开始增加
						{
							isend = true;
						
							isstart = false;
							isend = false;
							start_val1 = 0;

							// 更新当前找到的行
							Rect r = rects.get(rects.size() - 1);
							r.height = j - startj;
							rects.remove(rects.size() - 1);
							rects.add(r);
							
							times=0;//归0
							
						} else {
						
							//重新开始计算
							isstart=false;
							times = 0;// 相差过大，重新开始
							
							
							Rect r = rects.get(rects.size() - 1);
							if(r.height<50)
								rects.remove(rects.size() - 1);
							
							
							/*
							// 找到开始
							if (j > 100 &&j<imgheight-100&& start_val1 / start_val2 > 1 && start_val2 < max * 2 / 3) {

							
							times = 0;// 相差过大，重新开始
							startv=num;
							isstart = true;
							startj = j;
							// 从新寻找
							Rect r = rects.get(rects.size() - 1);
							rects.remove(rects.size() - 1);
							rects.add(r);
							}
							continue;*/
						}

					}
				}

			}

		}
		
		Rect r = rects.get(rects.size() - 1);
		if(r.height<50)
			rects.remove(rects.size() - 1);
		
		
		//特殊处理 对于拍照无边框，比较贴近的，有多个计算结果的，直接合并所有区域
		if(rects.size()>1)
		{
			Rect rall=new Rect(rects.get(0).x,rects.get(0).y,rects.get(0).width,rects.get(rects.size()-1).y+rects.get(rects.size()-1).height-rects.get(0).y);
			
			rects.clear();
			rects.add(rall);
		}
		
		
		

		return rects;

	}

	@FXML
	private void ocrSimple(Mat pic) {

		// File imageFile = new File("eurotext.tif");
		ITesseract instance = new Tesseract(); // JNA Interface Mapping
		// ITesseract instance = new Tesseract1(); // JNA Direct Mapping
		// File tessDataFolder = LoadLibs.extractTessResources("tessdata"); // Maven
		// build bundles English data
		File tessDataFolder = new File("F:\\Program Files (x86)\\Tesseract-OCR\\tessdata");
		instance.setDatapath(tessDataFolder.getPath());
		instance.setLanguage("chi_sim");

		BufferedImage bimage_id = (BufferedImage) HighGui.toBufferedImage(pic);

		String id = "error";
		try {
			id = instance.doOCR(bimage_id);
		} catch (TesseractException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		// isok = Imgcodecs.imwrite(f.getAbsolutePath() + "\\" + "id" + ".jpg", m_id);

		String output = id;// "name:" + name + "\r\n";
		// output += "addr:" + addr + "\r\n";
		// output += "id:" + id + "\r\n";
		System.out.println(output);

		transformedImage.setVisible(false);
		txtinfo.setText(output);

	}

	/**
	 * 二值化选择边框，计算身份证
	 * 
	 * @author zj
	 * @date 2018年8月9日
	 */
	@FXML
	private void cancueIdCardRage() {

		Mat m = new Mat();
		Mat filtered = new Mat();
		Mat thresholdImg = new Mat();
		Mat dilated_edges = new Mat();

		this.grayimage.copyTo(m);

		m.copyTo(filtered);
		// 滤波，模糊处理，消除某些背景干扰信息
		Imgproc.blur(m, filtered, new Size(1, 1));

		// 腐蚀操作，消除某些背景干扰信息
		// Imgproc.erode(filtered, filtered, new Mat(), new Point(-1, -1), 1);// 1, 1);

		// double maxgrayval = getMaxthresh(m);
		double maxgrayval = 40;// getthresh(m);
		System.out.println("maxgrayval:" + maxgrayval);

		// Imgproc.equalizeHist(filtered, filtered);

		// int thresh_type = Imgproc.THRESH_BINARY_INV; //反转
		int thresh_type = Imgproc.THRESH_OTSU;// 前后背景区分
		// threshValue
		// 双值化图像
		double to = 255;
		Imgproc.threshold(filtered, thresholdImg, maxgrayval * 3 / 4, to, thresh_type);

		Mat rectM = new Mat();
		this.image.copyTo(rectM);
		Mat rectM2 = new Mat();
		this.image.copyTo(rectM2);

		// 变换参数
		Mat transMat = new Mat();

		// 截取证件区域数据并进行透视变换为 垂直视角
		Mat realRect = new Mat();
		// Imgproc. warpPerspective(new Mat(rectM2, r),realRect,transMat,new
		// Size(r.width,r.height), Imgproc.INTER_LINEAR + Imgproc.WARP_INVERSE_MAP);

		Rect r = doContours(thresholdImg, rectM, transMat);
		if (r.width == 0 && r.height == 0) {
			// 轮廓计算失败
			r.width = filtered.width();

			r.height = filtered.height();

			rectM2.copyTo(realRect);
		} else {
			// 轮廓ok,变换原图

			Imgproc.warpPerspective(rectM2, realRect, transMat, rectM2.size(), Imgproc.INTER_LINEAR); // +
																										// Imgproc.WARP_INVERSE_MAP
																										//
		}

		showImg(originalImage3, realRect);

		Imgproc.cvtColor(thresholdImg, thresholdImg, Imgproc.COLOR_GRAY2BGR);
		Imgproc.rectangle(rectM, r.tl(), r.br(), new Scalar(0, 0, 255), 2);

		showImg(originalImage2, rectM);

		doCardReg(new Mat(realRect, r));

	}

	/**
	 * 对二值化后的身份证图片 ，定位行、字段等 区域原图,二值化的图
	 * 
	 * @param thresholdImg
	 * @author zj
	 * @date 2018年8月7日
	 */
	private void cacul2ValMat(Mat rectM, Mat thresholdImg) {
		// savetoImg(thresholdImg, "thresholdImg");

		// Rect r = doContours(thresholdImg, rectM);

		// savetoImg(rectM, "rectM");

		/*
		 * if (r.height == 0 && r.width == 0) { r.height = thresholdImg.height();
		 * r.width = thresholdImg.width();
		 * 
		 * }
		 */

		int linewidth = (int) Math.round(((double) rectM.width()) / 500.0);

		Mat fMat = new Mat(thresholdImg, new Rect(0, 0, thresholdImg.width(), thresholdImg.height()));

		Mat fRMat = new Mat();
		rectM.copyTo(fRMat);

		// savetoImg(fMat, "gray_rect");
		// savetoImg(fRMat, "rectM_rect");

		showImg(this.transformedImage2, fMat);

		Mat yhist = new Mat();
		List<Rect> rects = cany(fMat, yhist);

		showImg(this.transformedImage, yhist);

		Imgproc.cvtColor(fMat, fMat, Imgproc.COLOR_GRAY2BGR);
		List<Mat> addrs = new ArrayList<>();
		double addr_width = 0;
		double addr_x = 0;
		for (int i = 0; i < rects.size(); i++) {
			// if(rects.get(i).height>30)
			// {

			Rect yrect = rects.get(i);
			System.out.println(i + ": x:" + rects.get(i).x + " y:" + rects.get(i).y + " height:" + rects.get(i).height);
			Imgproc.rectangle(fRMat, rects.get(i).tl(), rects.get(i).br(), new Scalar(255, 0, 255), linewidth);

			// Mat rowMat=new Mat(fMat,new Rect(yrect.x,yrect.y, yrect.width*5/8,
			// yrect.height));
			Mat rowMat = new Mat(fMat, new Rect(yrect.x, yrect.y, yrect.width * 5 / 8, yrect.height));
			if (i == rects.size() - 1)
				rowMat = new Mat(fMat, new Rect(yrect.x, yrect.y, yrect.width, yrect.height));

			Mat xhist = new Mat();
			List<Rect> xrects = canx(rowMat, xhist);

			if (i == 2)
				showImg(this.antitransformedImage2, xhist);
			// else
			// showImgHalf(this.antitransformedImage2, xhist);

			// System.out.println("-------------");

			double paramsize = 10;// 区域扩大制定像素

			List<Mat> years = new ArrayList<>();
			List<Mat> nations = new ArrayList<>();

			for (int j = 0; j < xrects.size(); j++) {
				Rect xrect = xrects.get(j);

				// System.out.println(j + ": x:" + xrect.x + " y:" + xrect.y + " width:" +
				// xrect.width);

				double x = xrect.x - paramsize < 0 ? 0 : xrect.x - paramsize;
				double y = yrect.y - paramsize < 0 ? 0 : yrect.y - paramsize;

				// 每一行的身份证字段区域
				if (i == 1 || i == 2) {
					Imgproc.rectangle(fRMat, new Point(x, y),
							new Point(xrect.x + xrect.width + paramsize, yrect.y + yrect.height + paramsize),
							new Scalar(0, 255, 0), linewidth);

					if (i == 1) {
						Mat mline = new Mat(rectM, new Rect(new Point(x, y),
								new Point(xrect.x + xrect.width + paramsize, yrect.y + yrect.height + paramsize)));
						if (j == 0)
							RuihuaAndsavetoImg(mline, "sex");
						else if (j == xrects.size() - 1) {
							mline = new Mat(rectM,
									new Rect(new Point(xrects.get(1).x - paramsize, yrect.y - paramsize),
											new Point(
													xrects.get(xrects.size() - 1).x
															+ xrects.get(xrects.size() - 1).width + paramsize,
													yrect.y + yrect.height + paramsize)));

							RuihuaAndsavetoImg(mline, "nation");
						}

					} else if (i == 2) {
						Mat mline = new Mat(rectM, new Rect(new Point(x, y),
								new Point(xrect.x + xrect.width + paramsize, yrect.y + yrect.height + paramsize)));

						if (j == xrects.size() - 2)
							RuihuaAndsavetoImg(mline, "month");
						else if (j == xrects.size() - 1) {
							RuihuaAndsavetoImg(mline, "day");

							mline = new Mat(rectM,
									new Rect(new Point(xrects.get(0).x - paramsize, yrect.y - paramsize),
											new Point(
													xrects.get(xrects.size() - 3).x
															+ xrects.get(xrects.size() - 3).width + paramsize,
													yrect.y + yrect.height + paramsize)));

							RuihuaAndsavetoImg(mline, "year");
						}

					}

				}

			}

			if (xrects.size() > 0) {
				double x = xrects.get(0).x - paramsize < 0 ? 0 : xrects.get(0).x - paramsize;
				double y = yrect.y - paramsize < 0 ? 0 : yrect.y - paramsize;

				// 根据不同行，处理不同的区域合并问题 其他行合并显示
				if (i != 1 && i != 2)
					Imgproc.rectangle(fRMat, new Point(x, y),
							new Point(xrects.get(xrects.size() - 1).x + xrects.get(xrects.size() - 1).width + paramsize,
									yrect.y + yrect.height + paramsize),
							new Scalar(255, 255, 0), linewidth);

				Mat mline = new Mat(rectM,
						new Rect(new Point(x, y), new Point(
								xrects.get(xrects.size() - 1).x + xrects.get(xrects.size() - 1).width + paramsize,
								yrect.y + yrect.height + paramsize)));

				if (i == 3) {
					addr_x = x;
					addr_width = xrects.get(xrects.size() - 1).x + xrects.get(xrects.size() - 1).width + paramsize;
				}

				if (i == 0)
					RuihuaAndsavetoImg(mline, "name");
				else if (i == (rects.size() - 1))
					RuihuaAndsavetoImg(mline, "id");
				else if (i != 1 && i != 2) {
					mline = new Mat(rectM,
							new Rect(new Point(addr_x, y), new Point(addr_width, yrect.y + yrect.height + paramsize)));
					addrs.add(mline);
				}

			}
			// System.out.println("===========================");

			// }
		}

		try {
			Mat addr = new Mat();
			Core.vconcat(addrs, addr);
			RuihuaAndsavetoImg(addr, "addrs");

			/*
			 * Imgproc.cvtColor(addr, addr, Imgproc.COLOR_BGR2GRAY);
			 * Imgproc.equalizeHist(addr, addr);
			 */
			// addr = blacTwovalue2(addr, 65);
			// Imgproc.blur(addr, addr, new Size(3, 3));

			// RuihuaAndsavetoImg(addr, "addrs_new");

		} catch (Exception e) {
			System.out.println("addr failed");
		}

		showImg(this.antitransformedImage, fRMat);

		// Mat xhist =new Mat();
		// List<Rect> xrects= canx(fMat,xhist);

		// showImg(this.antitransformedImage2, xhist);

		readAndOcr();
	}

	private String ocr(Mat data) {

		String name = "";
		File imageFile = new File("eurotext.tif");
		ITesseract instance = new Tesseract(); // JNA Interface Mapping

		File tessDataFolder = new File("F:\\Program Files (x86)\\Tesseract-OCR\\tessdata");
		instance.setDatapath(tessDataFolder.getPath());
		instance.setLanguage("chi_sim+num");

		try {

			BufferedImage bimage_name = (BufferedImage) HighGui.toBufferedImage(data);
			name = instance.doOCR(bimage_name);

		} catch (Exception e) {
			System.err.println(e.getMessage());
		}
		return name;
	}

	private String ocrNum(Mat data) {

		String name = "";
		File imageFile = new File("eurotext.tif");
		ITesseract instance = new Tesseract(); // JNA Interface Mapping

		File tessDataFolder = new File("F:\\Program Files (x86)\\Tesseract-OCR\\tessdata");
		instance.setDatapath(tessDataFolder.getPath());
		instance.setLanguage("num");

		try {

			BufferedImage bimage_name = (BufferedImage) HighGui.toBufferedImage(data);
			name = instance.doOCR(bimage_name);

		} catch (Exception e) {
			System.err.println(e.getMessage());
		}
		return name;
	}

	/**
	 * 单纯计算y投影。
	 * 
	 * @param input
	 *            ,output, 直接显示的图形
	 * @return 单独一维数据mat
	 * @author zj
	 * @date 2019年3月5日
	 */
	private Mat canyStepOne(Mat input, Mat output, Double Maxnum) {

		// Mat output = new Mat();

		int imgheight = input.height();
		int imgwidth = input.width();
		// 存储y轴 左右投影
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
			// 计算最大的y投影，后续计算最大显示宽度使用
			double num = hist.get(0, i)[0];
			if (num > max)
				max = num;
		}

		Maxnum = max;

		// 显示y轴投影
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

		return hist;
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

		Double max = 0D;
		Mat hist = canyStepOne(input, output, max);

		// 计算纵向区域
		List<Rect> rects = new ArrayList<>();
		boolean isstart = false;

		double start_val1 = 0;// 前一个y值
		double start_val2 = 0;// 后一个y值
		int startj = 0; // 确定一行开始的y值
		double rate = 10;// 第二行超第一行的倍数
		double minrate = 0.01;

		int times = 0;
		int maxtimes = 5;// 像素值突然增大为区域开始
		boolean isend = false;
		for (int j = 0; j < imgheight; j++) {
			double num = hist.get(0, j)[0];

			if (!isstart) {
				// 顶部和底部的排除
				if (num > 0 && j > 20 && j < imgheight - 20) {
					if (start_val1 == 0)
						start_val1 = num;
					else {
						start_val2 = num;

						// 找到开始
						if (start_val2 / start_val1 >= 1.4 && (start_val2 > 10)) {
							isstart = true;

						} else {
							// 重新开始计算 第一个y
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

				if (num >= 0) {
					if (start_val1 == 0)
						start_val1 = num;
					else {
						start_val2 = num;

						// 找到结束，每一行至少>3%高度
						if (start_val1 / start_val2 > 1.4 && (j - startj > imgheight * 0.035) && start_val2 < 15) {
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
					isend = false;
					start_val1 = 0;

					// 更新当前找到的行
					Rect r = rects.get(rects.size() - 1);
					r.height = j - startj;
					rects.remove(rects.size() - 1);
					rects.add(r);
				}

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
	private List<Rect> canx(Mat input, Mat output) {

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

				if (input.get(i, j)[0] == 0) {

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
			v2[j] = 0;
		}
		hist2.put(0, 0, v2);
		for (int j = 0; j < imgwidth; j++) {

			double y = (hist.get(0, j)[0] / max) * imgheight;

			// B component or gray image
			Imgproc.line(hist2, new Point(j, imgheight), new Point(j, y), new Scalar(255, 0, 0), 2, 8, 0);
		}

		// Imgproc.cvtColor(hist2, hist2, Imgproc.COLOR_GRAY2BGR);
		hist2.copyTo(output);

		// rects
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
		for (int j = 0; j < imgwidth; j++) {
			double num = hist.get(0, j)[0];

			if (!isstart) {
				// 边框上的不检测
				if (num > 0 && j >= 20 && j <= imgwidth - 30) {
					if (start_val1 == 0)
						start_val1 = num;
					else {
						start_val2 = num;

						// 找到开始
						if (start_val2 / start_val1 >= 1 && start_val2 > 5) {
							isstart = true;

						} else {
							// 重新开始计算
							start_val1 = num;
						}
					}

				}

				if (isstart) {
					startj = j;
					Rect r = new Rect(j, 0, 0, imgheight);

					rects.add(r);

				}

			} else {

				if (num >= 0) {
					if (start_val1 == 0)
						start_val1 = num;
					else {
						start_val2 = num;

						if (start_val2 <= max * 0.01)
							times++;

						// 找到结束
						// if( (start_val1 / start_val2 > 1.4 && (j - startj >imgwidth*0.031) &&
						// start_val2 < 1)
						if ((start_val1 / start_val2 > 1.4 && (j - startj > 2) && start_val2 < 1)
						/* ||(times>=maxtimes) */
						) {
							System.out.println("j - startj :" + (j - startj) + " /imgwidth*0.031：" + imgwidth * 0.031
									+ "/ start_val2:" + start_val2);

							isend = true;
							start_val1 = 0;
							times = 0;
						} else {
							// 重新开始计算
							start_val1 = num;
						}

					}

				}

				if (isend) {

					isstart = false;
					isend = false;
					start_val1 = 0;

					Rect r = rects.get(rects.size() - 1);
					r.width = j - startj;
					rects.remove(rects.size() - 1);
					rects.add(r);
				}

			}

		}

		return rects;

	}

	private void showImg(ImageView view, Mat m) {
		// show the image
		this.updateImageView(view, Utils.mat2Image(m));
		// set a fixed width
		view.setFitWidth(450);

		view.setFitHeight(350);

	}

	private void showImgHalf(ImageView view, Mat m) {
		// show the image
		this.updateImageView(view, Utils.mat2Image(m));
		// set a fixed width
		view.setFitWidth(450 / 2);

		view.setFitHeight(350 / 2);

	}

	/**
	 * candy二值化选择边框，计算身份证 区域识别
	 * 
	 * @author zj
	 * @date 2018年8月6日
	 */
	@FXML
	private void canculRangeByCandy() {

		Mat m = new Mat();
		Mat filtered = new Mat();
		Mat edges = new Mat();
		Mat dilated_edges = new Mat();
		this.image.copyTo(m);

		// 滤波，模糊处理，消除某些背景干扰信息
		Imgproc.blur(m, filtered, new Size(3, 3));

		// savetoImg(filtered, "blur_33");

		// 腐蚀操作，消除某些背景干扰信息
		Imgproc.erode(filtered, filtered, new Mat(), new Point(-1, -1), 3);// 1, 1);

		// savetoImg(filtered, "blur_33_erode");

		// 获取直方图均衡化后的灰度像素均值.
		double mean = getAvGray(m);

		// double thresh =mean;
		double thresh = this.threshold.getValue();// 25D;

		String valuesToPrint = "threshValue: " + this.threshold.getValue();
		Utils.onFXThread(this.hsvValuesProp, valuesToPrint);

		// this.grayimage.copyTo(filtered);

		// mean=194.49;

		/*
		 * double threshmin=mean*0.44; double threshmax=mean*1.33;
		 */

		// 奥巴马头像检查 17-17*3
		double threshmin = thresh;
		double threshmax = threshmin * 3;

		System.out.println("*****mean:" + mean + "/threshmin:" + threshmin + "/threshmax:" + threshmax);

		// 图像线条化处理
		Imgproc.Canny(filtered, edges, threshmin, threshmax);

		savetoImg(edges, "blur_33_erode_edges");

		// show the image
		// showImg(antitransformedImage, edges);

		// 膨胀操作，尽量使边缘闭合，使图片最外层的边框变得明显，方便后续从线图中取出最外层的边框。
		Imgproc.dilate(edges, dilated_edges, new Mat(), new Point(-1, -1), 3);// , 1, 1);

		Mat transMat = new Mat();
		dilated_edges.copyTo(transMat);

		Mat draM = new Mat();
		this.image.copyTo(draM);

		Rect r = doContours(dilated_edges, draM, transMat);

		Mat rectM2 = new Mat();
		this.image.copyTo(rectM2);

		Mat realRect = new Mat();
		if (r.width == 0 && r.height == 0) {
			// 轮廓计算失败
			r.width = dilated_edges.width();

			r.height = dilated_edges.height();

			rectM2.copyTo(realRect);
		} else {
			// 轮廓ok,变换原图
			Imgproc.warpPerspective(rectM2, realRect, transMat, rectM2.size(),
					Imgproc.INTER_LINEAR + Imgproc.WARP_INVERSE_MAP);
		}

		showImg(originalImage3, realRect);

		Imgproc.cvtColor(dilated_edges, dilated_edges, Imgproc.COLOR_GRAY2BGR);
		Mat stepM = new Mat();
		dilated_edges.copyTo(stepM);

		Imgproc.rectangle(draM, r.tl(), r.br(), new Scalar(0, 0, 255), 2);

		showImg(originalImage2, draM);

		Mat idcarmat = new Mat(realRect, r);
		// 放大
		if (idcarmat.width() < 500)
			Imgproc.pyrUp(idcarmat, idcarmat);

		doCardReg(idcarmat);

	}

	/**
	 * 绘制 -轮廓，返回轮廓矩形 , 并且将原图矩形变换较证
	 * 
	 * @param dilated_edges
	 *            输入
	 * @param outContouMat
	 *            画上轮廓的输出
	 * @param transMat
	 *            透视变换mat
	 * @return
	 * @author zj
	 * @date 2018年8月9日
	 */
	private Rect doContours(Mat dilated_edges, Mat outContouMat, Mat transMat) {

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
						Imgproc.drawContours(outContouMat, contours, i, new Scalar(255, 255, 0), 3);

						MatOfPoint2f rectMat_dest = new MatOfPoint2f(
								new Point[] { new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y),
										new Point(rect.x + rect.width, rect.y + rect.height),
										new Point(rect.x, rect.y + rect.height), });

						// 确定 approxCurve 点的顺序
						List<Point> pts = new ArrayList<>();
						Point pt1 = new Point(approxCurve.get(0, 0));
						Point pt2 = new Point(approxCurve.get(1, 0));
						Point pt3 = new Point(approxCurve.get(2, 0));
						Point pt4 = new Point(approxCurve.get(3, 0));
						pts.add(pt1);
						pts.add(pt2);
						pts.add(pt3);
						pts.add(pt4);
						Point[] npts = sortRectPoints(pts);

						for (Point point : npts) {
							// 绘制四个订点
							Imgproc.drawMarker(outContouMat, point, new Scalar(255, 0, 0), Imgproc.MARKER_SQUARE, 20, 5,
									8);

						}

						MatOfPoint2f rectMat_src = new MatOfPoint2f(new Point[] { npts[0], npts[1], npts[2], npts[3] });

						// 矩形变换 获取 规则->不规则
						Mat tp = Imgproc.getPerspectiveTransform(rectMat_dest, rectMat_src);
						tp.copyTo(transMat);
						
						
						//只计算第一个有效矩形
						//只计算第一个找到的矩形
						//锁匠IMEI识别测试
						break;

						// 使用变换
						// Imgproc. warpPerspective(dilated_edges,outtransMat,tp,dilated_edges.size(),
						// Imgproc.INTER_LINEAR + Imgproc.WARP_INVERSE_MAP);

					}
				}
			}
			
			
			//break;
			
		}

		// savetoImg(dilated_edges, "contour");

		return rect;

	}

	/**
	 * 凸四边形顶点排序，左上角开始顺时针排序
	 * 
	 * @param pts
	 * @return
	 * @author zj
	 * @date 2018年8月9日
	 */
	private Point[] sortRectPoints(List<Point> pts) {
		// List<Point> rst = new ArrayList();

		if (pts.size() != 4)
			return null;

		Point[] apts = new Point[4];

		double xtotal = 0;
		double ytotal = 0;
		double xav = 0;
		double yav = 0;
		for (Point point : pts) {
			xtotal += point.x;
			ytotal += point.y;
		}
		// 中心点
		xav = xtotal / pts.size();
		yav = ytotal / pts.size();

		for (Point point : pts) {

			if (point.x < xav && point.y < yav)
				apts[0] = point;
			else if (point.x > xav && point.y < yav)
				apts[1] = point;
			else if (point.x > xav && point.y > yav)
				apts[2] = point;
			else if (point.x < xav && point.y > yav)
				apts[3] = point;
		}

		return apts;

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
			this.image = Imgcodecs.imread(file.getAbsolutePath(), Imgcodecs.IMREAD_COLOR);
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

			showImg(transformedImage3, this.grayimage);
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

	private void RuihuaAndsavetoImg(Mat m, String name) {

		String path = FourierController.class.getResource("/").getPath();
		File f1 = new File(path + "\\out\\ori\\");
		File f = new File(path + "\\out\\ids\\");
		if (!f.exists())
			f.mkdirs();

		if (!f1.exists())
			f1.mkdirs();

		Imgcodecs.imwrite(f1.getAbsolutePath() + "\\" + name + ".png", m);

		Imgproc.pyrUp(m, m);
		Imgproc.pyrUp(m, m);
		Imgproc.pyrUp(m, m);
		Mat structElement1 = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(2, 2), new Point(-1, -1));
		/*
		 * Mat structElement2 = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new
		 * Size(1,1), new Point(-1, -1));
		 */
		Imgproc.erode(m, m, structElement1, new Point(-1, -1), 3);// 1, 1);

		Imgproc.dilate(m, m, structElement1, new Point(-1, -1), 3);

		Imgcodecs.imwrite(f.getAbsolutePath() + "\\" + name + "_b.png", m);
		// m = BackgroundRemoval(m);

		// if(name.equals("id")) {
		// double delval = avLight(m);

		// m = duibidu(m);

		Imgproc.blur(m, m, new Size(1, 1));
		m = getSimpleImgForJashiz(m, 0, 0);

		// showImg(antitransformedImage3, m);
		// m=twoValueIdCard(m);

		Imgproc.pyrDown(m, m);

		m = ruihua(m);

		// Imgproc.cvtColor(m,m, Imgproc.COLOR_BGR2GRAY);

		// }

		Imgcodecs.imwrite(f.getAbsolutePath() + "\\" + name + ".png", m);

	}

	private void readAndOcr() {

		String path = FourierController.class.getResource("/").getPath();
		File f = new File(path + "\\out\\ids\\");
		for (File fs : f.listFiles()) {

			try {

				Mat m = Imgcodecs.imread(fs.getAbsolutePath());
				if (fs.getName().contains("id"))
					System.out.println(fs.getName() + ":" + ocrNum(m));
				else
					System.out.println(fs.getName() + ":" + ocr(m));

			} catch (Exception e) {
				continue;
			}

		}

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
		// optionally reorder the o quadrants of the magnitude image
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
