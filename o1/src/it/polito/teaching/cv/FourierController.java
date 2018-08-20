package it.polito.teaching.cv;

import javafx.scene.control.TextArea;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import javax.imageio.ImageIO;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.objdetect.Objdetect;

import it.polito.elite.teaching.cv.utils.Utils;
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
import net.sourceforge.tess4j.ITesseract;
import net.sourceforge.tess4j.Tesseract;
import net.sourceforge.tess4j.util.LoadLibs;

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
	private ImageView maskImage;

	@FXML
	private ImageView antitransformedImage;
	// a FXML button for performing the transformation
	@FXML
	private Button transformButton;
	// a FXML button for performing the antitransformation
	@FXML
	private Button antitransformButton;

	@FXML
	private Slider threshold;

	@FXML
	private Slider threshold2;

	@FXML
	private Slider threshold3;

	@FXML
	private Slider threshold4;

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

	private Mat orimage;

	private List<Mat> planes;
	// the final complex image
	private Mat complexImage;
	@FXML
	private Label hsvCurrentValues;

	@FXML
	private TextArea txtinfo;

	@FXML
	private Button doCannyBtn;
	// property for object binding
	private ObjectProperty<String> hsvValuesProp;

	/**
	 * Get the average hue value of the image starting from its Hue channel
	 * histogram
	 * 
	 * @param hsvImg
	 *            the current frame in HSV
	 * @param hueValues
	 *            the Hue component of the current frame
	 * @return the average Hue value
	 */
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
		return average = average / hsvImg.size().height / hsvImg.size().width;
	}

	// face cascade classifier
	private CascadeClassifier faceCascade;
	private int absoluteFaceSize;

	@FXML
	private void doBackgroundRemovalAbsDiff() {
		Mat currFrame = new Mat();
		image.copyTo(currFrame, this.image);

		Mat cmat = canny(currFrame);

		// Mat rst = BackgroundRemoval(currFrame);
		Mat rst = BackgroundRemovalAbsDiff(currFrame);

		// show the partial output
		this.updateImageView(this.transformedImage, Utils.mat2Image(rst));
		this.transformedImage.setFitWidth(850);

	}

	Point clickedPoint = new Point(0, 0);

	private Mat doBackgroundRemovalFloodFill(Mat frame) {

		Scalar newVal = new Scalar(255, 255, 255);
		Scalar loDiff = new Scalar(50, 50, 50);
		Scalar upDiff = new Scalar(50, 50, 50);
		Point seedPoint = clickedPoint;
		Mat mask = new Mat();
		Rect rect = new Rect();

		// Imgproc.floodFill(frame, mask, seedPoint, newVal);
		Imgproc.floodFill(frame, mask, seedPoint, newVal, rect, loDiff, upDiff, Imgproc.FLOODFILL_FIXED_RANGE);

		return frame;
	}

	private Mat BackgroundRemovalAbsDiff(Mat currFrame) {
		Mat greyImage = new Mat();
		Mat foregroundImage = new Mat();

		// if (oldFrame == null)
		Mat oldFrame = currFrame;

		Core.absdiff(currFrame, oldFrame, foregroundImage);
		// Imgproc.cvtColor(foregroundImage, greyImage, Imgproc.COLOR_BGR2GRAY);

		int thresh_type = Imgproc.THRESH_BINARY_INV;
		// if (this.inverse.isSelected())
		// thresh_type = Imgproc.THRESH_BINARY;

		Imgproc.threshold(greyImage, greyImage, 10, 255, thresh_type);
		currFrame.copyTo(foregroundImage, greyImage);

		oldFrame = currFrame;
		return foregroundImage;

	}

	private Mat BackgroundRemoval(Mat frame) {

		// init
		Mat hsvImg = new Mat();
		List<Mat> hsvPlanes = new ArrayList<>();
		Mat thresholdImg = new Mat();

		int thresh_type = Imgproc.THRESH_BINARY_INV;
		/*
		 * if (this.inverse.isSelected()) thresh_type = Imgproc.THRESH_BINARY;
		 */

		// threshold the image with the average hue value

		hsvImg.create(frame.size(), CvType.CV_8U);
		Mat ori = new Mat();
		Imgproc.cvtColor(frame, ori, Imgproc.COLOR_GRAY2BGR);
		Imgproc.cvtColor(this.orimage, hsvImg, Imgproc.COLOR_BGR2HSV);

		Core.split(hsvImg, hsvPlanes);
		double threshValue = this.getHistAverage(hsvImg, hsvPlanes.get(0));
		System.out.println("av threshValue:" + threshValue);
		/*
		 * double threshValue = this.threshold.getValue();// this.getHistAverage(hsvImg,
		 * hsvPlanes.get(0));
		 * 
		 * double to=threshold2.getValue();
		 * 
		 * int s3=(int)threshold3.getValue(); int s4=(int)threshold4.getValue();
		 */

		 threshValue = (int)threshold3.getValue(); //140;//
		// this.getHistAverage(hsvImg, hsvPlanes.get(0));

		System.out.println("threshValue:" + threshValue);
		double to = 180;// threshold2.getValue(); //10;//
		System.out.println("to:" + to);

		int s3 = 3;// (int)threshold3.getValue();
		int s4 = 3;// (int)threshold4.getValue();

		Imgproc.threshold(frame, thresholdImg, threshValue, to, thresh_type);

		Imgproc.blur(thresholdImg, thresholdImg, new Size(5, 5));

		// dilate to fill gaps, erode to smooth edges
		Imgproc.dilate(thresholdImg, thresholdImg, new Mat(), new Point(-1, -1), s3);
		Imgproc.erode(thresholdImg, thresholdImg, new Mat(), new Point(-1, -1), s4);

		Imgproc.threshold(thresholdImg, thresholdImg, threshValue, to, thresh_type);

		// create the new image
		Mat foreground = new Mat(frame.size(), CvType.CV_8UC3, new Scalar(255, 255, 255));
		frame.copyTo(foreground, thresholdImg);

		// show the current selected HSV range
		String valuesToPrint = "threshValue: " + threshValue + "- to：" + to + " s3:" + s3 + "-  s4：" + s4;
		Utils.onFXThread(this.hsvValuesProp, valuesToPrint);

		return foreground;
	}

	@FXML
	private void doBackgroundRemoval() {

		Mat src2 = new Mat();
		this.image.copyTo(src2);
		// Mat blurredImage = new Mat();
		// Imgproc.blur(src2, blurredImage, new Size(1, 1));
		Mat matc = new Mat();
		// Imgproc.cvtColor(src2, matc, Imgproc.COLOR_GRAY2BGR);

		avLight(this.orimage);

		String path = FourierController.class.getResource("/").getPath();
		File f = new File(path + "\\out");
		if (!f.exists())
			f.mkdirs();

		int imgwidth = src2.width();
		int imgheight = src2.height();

		try {

			matc = BackgroundRemoval(src2);
			
			//画轮廓
			//findContours(matc);
			
			//画直线
			findline(matc);

			boolean isok = Imgcodecs.imwrite(f.getAbsolutePath() + "\\" + "backremove.jpg", matc);

		} catch (Exception e) {
			// TODO: handle exception
		}

		//// show the partial output
		//this.updateImageView(this.transformedImage, Utils.mat2Image(matc));
	//	this.transformedImage.setFitWidth(450);
	}

	@FXML
	private void rage() {

		Mat frame = new Mat();
		image.copyTo(frame, this.image);

		Mat blurredImage = new Mat();
		Mat hsvImage = new Mat();
		Mat mask = new Mat();
		Mat morphOutput = new Mat();

		/*
		 * // remove some noise Imgproc.blur(frame, blurredImage, new Size(7, 7));
		 * 
		 * // convert the frame to HSV Imgproc.cvtColor(blurredImage, hsvImage,
		 * Imgproc.COLOR_BGR2HSV);
		 */

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

		// this.updateImageView(this.maskImage, Utils.mat2Image(mask));
		// this.maskImage.setFitHeight(350);

		double to = threshold2.getValue();

		double s3 = threshold3.getValue();
		int s4 = (int) threshold4.getValue();

		Mat line = new Mat();
		Mat matc = new Mat();
		Imgproc.cvtColor(mask, matc, Imgproc.COLOR_GRAY2BGR);

		Mat matcgray = new Mat();
		matc.copyTo(matcgray, matc);

		Imgproc.HoughLinesP(mask, line, 1, Math.PI / 180, 40, 35, 2.5);

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

				Imgproc.line(matcgray, ps, pend, new Scalar(0, 0, 0), 1);

				Imgproc.line(matc, ps, pend, new Scalar(0, 0, 255), 1);
				Imgproc.drawMarker(matc, ps, new Scalar(0, 255, 0), Imgproc.MARKER_DIAMOND, 20, 5, 8);
				Imgproc.drawMarker(matc, pend, new Scalar(255, 0, 0), Imgproc.MARKER_SQUARE, 20, 5, 8);
			} catch (Exception e) {
				e.printStackTrace();
			}
		}

		double smallx = 0;
		double snum = 0;
		for (Map.Entry<Double, Integer> data : xnums.entrySet()) {

			if (data.getValue() > snum) {
				smallx = data.getKey();
				snum = data.getValue();
			}
		}

		double bigx = image.width();
		double bignum = 0;
		for (Map.Entry<Double, Integer> data : x2nums.entrySet()) {

			if (data.getValue() > bignum) {
				bigx = data.getKey();
				bignum = data.getValue();
			}
		}

		// 排序点，获取顶部附件的点及表格最外层的点
		List<double[]> useLine = new ArrayList();
		int index = 0;
		for (int i = 0; i < line.rows(); i++) {
			double[] pt = line.get(i, 0);
			if (pt[0] != smallx || pt[2] != bigx)
				continue;
			else {

				// useLine.reshape(1,index+1);
				useLine.add(pt);
				index++;
			}
		}
		for (int i = 0; i < useLine.size(); i++) {
			System.out.println(useLine.get(i)[0] + "," + useLine.get(i)[1] + "," + useLine.get(i)[2] + ","
					+ useLine.get(i)[3] + "");
		}
		System.out.println("-------");
		// 排序mat
		for (int i = 0; i < useLine.size(); i++) {

			double[] di = useLine.get(i);

			for (int j = i + 1; j < useLine.size(); j++) {
				double[] dj = useLine.get(j);

				if (di[1] > dj[1]) {
					double[] tp = di;
					di = dj;
					dj = tp;
					useLine.set(i, di);
					useLine.set(j, dj);
				}
			}

		}

		for (int i = 0; i < useLine.size(); i++) {
			System.out.println(useLine.get(i)[0] + "," + useLine.get(i)[1] + "," + useLine.get(i)[2] + ","
					+ useLine.get(i)[3] + "");
		}

		String path = FourierController.class.getResource("/").getPath();
		File f = new File(path + "\\out");
		if (!f.exists())
			f.mkdirs();

		double startx = 0;
		double starty = 0;
		double endx = 0;
		double endy = 0;
		// 切割图片
		for (int i = 0; i < useLine.size(); i++) {
			try {

				if (i != 0) {
					startx = useLine.get(i - 1)[0];
					starty = useLine.get(i - 1)[1];
				}

				endx = useLine.get(i)[2];
				endy = useLine.get(i)[3];

				Rect rt = new Rect(new Point(startx, starty), new Point(endx, endy));
				Mat m = new Mat(matcgray, rt);

				boolean isok = Imgcodecs.imwrite(f.getAbsolutePath() + "\\" + i + ".jpg", m);
				System.out.println(i + ".jpg" + (isok ? " is " : " not ") + " done!");
			} catch (Exception e) {
				e.getMessage();
			}
		}

		Imgcodecs.imwrite(f.getAbsolutePath() + "\\gray" + ".jpg", matcgray);

		System.out.println("切割文件输出目录：" + path + "\\out\\");

		// Imgproc.cvtColor(mask, mask, Imgproc.COLOR_GRAY2BGR);

		// show the partial output

		this.updateImageView(this.transformedImage, Utils.mat2Image(matc));
		this.transformedImage.setFitWidth(650);
		this.transformedImage.setFitHeight(450);

		// this.updateImageView(this.maskImage, Utils.mat2Image(matc));
		// this.maskImage.setFitWidth(650);

		/*
		 * 
		 * // init List<MatOfPoint> contours = new ArrayList<>(); Mat hierarchy = new
		 * Mat();
		 * 
		 * // find contours Imgproc.findContours(cmat, contours, hierarchy,
		 * Imgproc.RETR_CCOMP, Imgproc.CHAIN_APPROX_SIMPLE);
		 * 
		 * Mat frame=new Mat(); // if any contour exist... if (hierarchy.size().height >
		 * 0 && hierarchy.size().width > 0) { // for each contour, display it in blue
		 * for (int idx = 0; idx >= 0; idx = (int) hierarchy.get(0, idx)[0]) {
		 * Imgproc.drawContours(frame, contours, idx, new Scalar(250, 0, 0)); } }
		 * 
		 * 
		 * 
		 * // show the result of the transformation as an image
		 * this.updateImageView(transformedImage, Utils.mat2Image(frame)); // set a
		 * fixed width this.transformedImage.setFitWidth(650); // preserve image ratio
		 */

	}

	private void findContours(Mat cmat) {

		List<MatOfPoint> contours = new ArrayList<>();
		Mat hierarchy = new Mat();
		// find contours
		Imgproc.findContours(cmat, contours, hierarchy, Imgproc.RETR_CCOMP, Imgproc.CHAIN_APPROX_SIMPLE);

		Mat frame = new Mat();
		this.orimage.copyTo(frame);
		// if any contour exist...
		if (hierarchy.size().height > 0 && hierarchy.size().width > 0) { // for each contour, display it in blue
			for (int idx = 0; idx >= 0; idx = (int) hierarchy.get(0, idx)[0]) {
				Imgproc.drawContours(frame, contours, idx, new Scalar(250, 0, 0));
			}
		}
		
		  this.updateImageView(transformedImage, Utils.mat2Image(frame)); // set a
		   this.transformedImage.setFitWidth(650); // preserve image ratio

	}

	
	
	private void findline(Mat mask) {
		
		Mat matc=new Mat();
		this.orimage.copyTo(matc);
		
		Mat line = new Mat();
		Imgproc.HoughLinesP(mask, line, 1, Math.PI / 180, 40, 35, 20);

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

				//Imgproc.line(matcgray, ps, pend, new Scalar(0, 0, 0), 1);

				Imgproc.line(matc, ps, pend, new Scalar(0, 0, 255), 1);
				Imgproc.drawMarker(matc, ps, new Scalar(0, 255, 0), Imgproc.MARKER_DIAMOND, 20, 5, 8);
				Imgproc.drawMarker(matc, pend, new Scalar(255, 0, 0), Imgproc.MARKER_SQUARE, 20, 5, 8);
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
		
		
		  this.updateImageView(transformedImage, Utils.mat2Image(matc)); // set a
		   this.transformedImage.setFitWidth(650); // preserve image ratio

		
		

	}
	
	/**
	 * Apply Canny
	 * 
	 * @param frame
	 *            the current frame
	 * @return an image elaborated with Canny
	 */
	@FXML
	private void doCanny() {
		// Mat image = Imgcodecs.imread(location);

		// init
		Mat grayImage = this.image;

		// using Canny's output as a mask, display the result
		Mat dest = new Mat();
		image.copyTo(dest, image);

		dest = canny(dest);

		// show the result of the transformation as an image
		this.updateImageView(transformedImage, Utils.mat2Image(dest));
		// set a fixed width
		this.transformedImage.setFitWidth(650);
		// preserve image ratio
		// this.transformedImage.setPreserveRatio(true);

		// enable the button for performing the antitransformation
		// this.antitransformButton.setDisable(false);
		// disable the button for applying the dft
		// this.transformButton.setDisable(true);
	}

	private Mat canny(Mat grayImage) {
		Mat detectedEdges = new Mat();

		// convert to grayscale
		// Imgproc.cvtColor(frame, grayImage, Imgproc.COLOR_BGR2GRAY);

		// reduce noise with a 3x3 kernel
		Imgproc.blur(grayImage, detectedEdges, new Size(3, 3));

		// canny detector, with ratio of lower:upper threshold of 3:1
		Imgproc.Canny(detectedEdges, detectedEdges, this.threshold.getValue(), this.threshold.getValue() * 3);

		return detectedEdges;
	}

	private void dodetectAndDisplay() {

		Mat frame = new Mat();
		this.orimage.copyTo(frame, this.orimage);

		MatOfRect faces = new MatOfRect();
		Mat grayFrame = new Mat();

		// init
		// Mat grayFrame = this.image;

		// convert the frame in gray scale
		Imgproc.cvtColor(frame, grayFrame, Imgproc.COLOR_BGR2GRAY);
		// equalize the frame histogram to improve the result
		Imgproc.equalizeHist(grayFrame, grayFrame);

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
		this.transformedImage.setFitWidth(650);

	}

	@FXML
	private void faceDetect1() {
		this.faceCascade.load("resources/haarcascades/haarcascade_frontalface_alt.xml");
		dodetectAndDisplay();
	}

	@FXML
	private void faceDetect_body() {
		this.faceCascade.load("resources/haarcascades/haarcascade_fullbody.xml");
		dodetectAndDisplay();
	}

	@FXML
	private void faceDetect_lbp() {
		this.faceCascade.load("resources/lbpcascades/lbpcascade_frontalface.xml");
		dodetectAndDisplay();
	}

	private static final int W = 400;

	private void MyEllipse(Mat img, double angle) {
		int thickness = 2;
		int lineType = 8;
		int shift = 0;
		Imgproc.ellipse(img, new Point(W / 2, W / 2), new Size(W / 4, W / 16), angle, 0.0, 360.0, new Scalar(255, 0, 0),
				thickness, lineType, shift);
	}

	String atom_window = "Drawing 1: Atom";

	private void thresd() {
		/*
		 * cv::cvtColor(image, image_gray, cv::COLOR_BGR2GRAY);
		 * cv::threshold(image_gray, image_gray, g_threshVal, g_threshMax,
		 * cv::THRESH_BINARY);
		 */
	}

	@FXML
	private void testEclipse() {
		Mat atom_image = Mat.zeros(W, W, CvType.CV_8UC3);
		Mat rook_image = Mat.zeros(W, W, CvType.CV_8UC3);
		MyEllipse(atom_image, 90.0);
		MyEllipse(atom_image, 0.0);
		MyEllipse(atom_image, 45.0);
		MyEllipse(atom_image, -45.0);
		// MyFilledCircle( atom_image, new Point( W/2, W/2) );
		// MyPolygon( rook_image );
		Imgproc.rectangle(rook_image, new Point(0, 7 * W / 8), new Point(W, W), new Scalar(0, 255, 255), -1, 8, 0);
		// MyLine( rook_image, new Point( 0, 15*W/16 ), new Point( W, 15*W/16 ) );
		// MyLine( rook_image, new Point( W/4, 7*W/8 ), new Point( W/4, W ) );
		// MyLine( rook_image, new Point( W/2, 7*W/8 ), new Point( W/2, W ) );
		// MyLine( rook_image, new Point( 3*W/4, 7*W/8 ), new Point( 3*W/4, W ) );
		HighGui.imshow(atom_window, atom_image);
		// HighGui.moveWindow( atom_window, 0, 200 );
		// HighGui.imshow( rook_window, rook_image );
		// HighGui.moveWindow( rook_window, W, 200 );
		HighGui.waitKey(200);
		// System.exit(0);
	}

	/**
	 * 去除图片背景，突出文字 ，hsv调整，
	 * 
	 * @param frame
	 * @author zj
	 * @date 2018年8月2日
	 */
	private Mat getSimpleImg(Mat frame, double starth, double endh) {

		Mat mask = new Mat();

		Scalar minValues = new Scalar(starth, 0, 0);// this.saturationStart.getValue(),
		// this.valueStart.getValue());
		Scalar maxValues = new Scalar(endh, 255, 255);// this.saturationStop.getValue(),
		// this.valueStop.getValue());

		// show the current selected HSV range
		String valuesToPrint = "Hue range: " + minValues.val[0] + "-" + maxValues.val[0] + "\tSaturation range: "
				+ minValues.val[1] + "-" + maxValues.val[1] + "\tValue range: " + minValues.val[2] + "-"
				+ maxValues.val[2];
		Utils.onFXThread(this.hsvValuesProp, valuesToPrint);

		// threshold HSV image to select tennis balls
		Core.inRange(frame, minValues, maxValues, mask);

		return mask;
	}

	private Mat getSimpleImgByparams(Mat frame) {

		Mat mask = new Mat();

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

	// 锐化
	private Mat ruihua(Mat matc) {
		/*
		 * 自定义核 0 -1 0 -1 5 -1 0 -1 0
		 */

		Mat kernel = new Mat(3, 3, CvType.CV_16SC1);
		kernel.put(0, 0, 0, -1, 0, -1, 5, -1, 0, -1, 0);
		// 对图像和自定义核做卷积
		Imgproc.filter2D(matc, matc, matc.depth(), kernel);
		return matc;
	}

	/**
	 * 放大图片，
	 * 
	 * @param matc
	 * @param beishu
	 *            放大倍数
	 * @return
	 * @author zj
	 * @date 2018年8月3日
	 */
	private Mat pryUp(Mat matc, int beishu) {
		Mat big = new Mat();

		Imgproc.pyrUp(matc, big, new Size(matc.width() * beishu, matc.height() * beishu));

		return big;
	}

	/**
	 * 模糊边界，去噪
	 * 
	 * @param matc
	 * @return
	 * @author zj
	 * @date 2018年8月2日
	 */
	private Mat blur(Mat matc) {
		Mat blurredImage = new Mat();
		Imgproc.blur(matc, blurredImage, new Size(1, 1));

		return blurredImage;
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

		hsvImage = new Mat();
		Imgproc.cvtColor(tmp, hsvImage, Imgproc.COLOR_BGR2HLS);

		int imgheight = hsvImage.height();
		int imgwidth = hsvImage.width();

		// Imgcodecs.imwrite(f.getAbsolutePath() + "\\" + "hls.jpg", hsvImage);
		double total = 0;
		double avlight = 0;
		for (int i = 0; i < imgwidth - 1; i++) {
			for (int j = 0; j < imgheight; j++) {
				// int j=imgheight/2;
				double[] vals = hsvImage.get(j, i);
				double light = vals[1];
				total += light;
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
	 * 获取给定BGR图像的平均亮度
	 * 
	 * @param matc
	 * @return
	 * @author zj
	 * @date 2018年8月2日
	 */
	private int avLightForjiashi(Mat matc) {

		Mat tmp = new Mat();
		matc.copyTo(tmp, matc);
		// 计算图片平均亮度
		Mat hsvImage = new Mat();

		hsvImage = new Mat();
		Imgproc.cvtColor(tmp, hsvImage, Imgproc.COLOR_BGR2HLS);

		int imgheight = hsvImage.height();
		int imgwidth = hsvImage.width();

		// Imgcodecs.imwrite(f.getAbsolutePath() + "\\" + "hls.jpg", hsvImage);
		double total = 0;
		double avlight = 0;
		for (int i = 0; i < imgwidth - 1; i++) {
			for (int j = 0; j < imgheight; j++) {
				// int j=imgheight/2;
				double[] vals = hsvImage.get(j, i);
				double light = vals[1];
				total += light;
			}
		}
		avlight = total / ((double) (imgwidth * imgheight));

		System.out.println("平均亮度:" + avlight);

		int delval = 118;
		if (avlight > 150 && avlight <= 160)
			delval = 122;
		else if (avlight > 160 && avlight <= 170)
			delval = 135;
		else if (avlight > 170 && avlight <= 180)
			delval = 161;
		else if (avlight > 180)
			delval = 175;

		return delval;
	}

	/**
	 * 身份证识别
	 * 
	 * @author zj
	 * @date 2018年8月3日
	 */
	@FXML
	private void ocrTest() {
		Mat src2 = new Mat();
		this.image.copyTo(src2, this.image);
		// Mat blurredImage = new Mat();
		// Imgproc.blur(src2, blurredImage, new Size(1, 1));
		Mat matc = new Mat();
		Imgproc.cvtColor(src2, matc, Imgproc.COLOR_GRAY2BGR);

		String path = FourierController.class.getResource("/").getPath();
		File f = new File(path + "\\out");
		if (!f.exists())
			f.mkdirs();

		int imgwidth = matc.width();
		int imgheight = matc.height();

		double xstartrate = 0.163; // 其他字段开始x比例

		double xid_startrate = 0.334;// 身份证开始x比例

		double x_name_start = xstartrate;//
		double x_name_end = 0.36;// 姓名结束x
		double y_name_start = 0.11;// 姓名开始高度
		double y_name_end = 0.21;// 姓名开始高度

		double x_address_start = xstartrate;//
		double x_address_end = 0.62;// 地址结束x
		double y_address_start = 0.47;// 地址开始高度
		double y_address_end = 0.75;// 地址结束高度

		double x_id_start = xid_startrate;//
		double x_id_end = 0.92;// 身份证号结束x
		double y_id_start = 0.82;// 身份证号结束x
		double y_id_end = 0.9;// 身份证号结束y

		Rect rt_name = new Rect(new Point(x_name_start * imgwidth, y_name_start * imgheight),
				new Point(x_name_end * imgwidth, y_name_end * imgheight));
		Mat m_name = new Mat(matc, rt_name);

		Rect rt_addr = new Rect(new Point(x_address_start * imgwidth, y_address_start * imgheight),
				new Point(x_address_end * imgwidth, y_address_end * imgheight));
		Mat m_addr = new Mat(matc, rt_addr);

		Rect rt_id = new Rect(new Point(x_id_start * imgwidth, y_id_start * imgheight),
				new Point(x_id_end * imgwidth, y_id_end * imgheight));
		Mat m_id = new Mat(matc, rt_id);

		File imageFile = new File("eurotext.tif");
		ITesseract instance = new Tesseract(); // JNA Interface Mapping
		// ITesseract instance = new Tesseract1(); // JNA Direct Mapping
		// File tessDataFolder = LoadLibs.extractTessResources("tessdata"); // Maven
		// build bundles English data
		File tessDataFolder = new File("F:\\Program Files (x86)\\Tesseract-OCR\\tessdata");
		instance.setDatapath(tessDataFolder.getPath());
		instance.setLanguage("chi_sim");

		try {

			// String result = instance.doOCR(imageFile);

			int delval = 90; // hs 去背参数
			if (imgwidth > 1000) {

				delval = avLight(m_name);

				m_name = getSimpleImg(m_name, 0, delval);
				m_name = ruihua(m_name);
			} else {
				// m_name=ruihua(m_name);
			}

			BufferedImage bimage_name = (BufferedImage) HighGui.toBufferedImage(m_name);
			String name = instance.doOCR(bimage_name);
			boolean isok = Imgcodecs.imwrite(f.getAbsolutePath() + "\\" + "name.jpg", m_name);

			if (imgwidth > 1000) {
				delval = avLight(m_addr);
				m_addr = getSimpleImg(m_addr, 0, delval);
				m_addr = blur(m_addr);
			} else {
				// m_addr = getSimpleImg(m_addr, 0, 80);
				// m_addr=ruihua(m_addr);
			}
			// m_addr=ruihua(m_addr);

			BufferedImage bimage_addr = (BufferedImage) HighGui.toBufferedImage(m_addr);
			String addr = instance.doOCR(bimage_addr);
			isok = Imgcodecs.imwrite(f.getAbsolutePath() + "\\" + "addr" + ".jpg", m_addr);

			if (imgwidth > 1000) {
				delval = avLight(m_id);

				if (delval > 100) {

					m_id = getSimpleImg(m_id, 0, delval);

					m_id = blur(m_id);
				}
			} else {
				m_id = ruihua(m_id);
				/*
				 * m_id = getSimpleImg(m_id, 0,95); m_id=ruihua(m_id);
				 */

				// m_id=blur(m_id);
			}
			// m_id = getSimpleImg(m_id, 2, 85);
			// m_id=ruihua(m_id);

			BufferedImage bimage_id = (BufferedImage) HighGui.toBufferedImage(m_id);
			String id = instance.doOCR(bimage_id);
			isok = Imgcodecs.imwrite(f.getAbsolutePath() + "\\" + "id" + ".jpg", m_id);

			String output = "name:" + name + "\r\n";
			output += "addr:" + addr + "\r\n";
			output += "id:" + id + "\r\n";
			System.out.println(output);

			transformedImage.setVisible(false);
			txtinfo.setText(output);

		} catch (Exception e) {
			System.err.println(e.getMessage());
		}

		// boolean isok = Imgcodecs.imwrite(f.getAbsolutePath() + "\\" + i + ".jpg", m);

	}

	/**
	 * 驾驶证识别
	 * 
	 * @author zj
	 * @date 2018年8月3日
	 */
	@FXML
	private void ocrJashizhengTest() {
		Mat src2 = new Mat();
		this.image.copyTo(src2, this.image);
		// Mat blurredImage = new Mat();
		// Imgproc.blur(src2, blurredImage, new Size(1, 1));
		Mat matc = new Mat();
		Imgproc.cvtColor(src2, matc, Imgproc.COLOR_GRAY2BGR);

		String path = FourierController.class.getResource("/").getPath();
		File f = new File(path + "\\out");
		if (!f.exists())
			f.mkdirs();

		int imgwidth = matc.width();
		int imgheight = matc.height();

		double xstartrate = 0.132; // 其他字段开始x比例

		double xid_startrate = 0.405;// 身份证开始x比例

		double x_name_start = xstartrate;//
		double x_name_end = 0.424;// 姓名结束x
		double y_name_start = 0.255;// 姓名开始高度
		double y_name_end = 0.347;// 姓名开始高度

		double x_address_start = xstartrate;//
		double x_address_end = 0.898;// 地址结束x
		double y_address_start = 0.349;// 地址开始高度
		double y_address_end = 0.526;// 地址结束高度

		double x_id_start = xid_startrate;//
		double x_id_end = 0.837;// 身份证号结束x
		double y_id_start = 0.178;// 身份证号结束x
		double y_id_end = 0.251;// 身份证号结束y

		Rect rt_name = new Rect(new Point(x_name_start * imgwidth, y_name_start * imgheight),
				new Point(x_name_end * imgwidth, y_name_end * imgheight));
		Mat m_name = new Mat(matc, rt_name);

		Rect rt_addr = new Rect(new Point(x_address_start * imgwidth, y_address_start * imgheight),
				new Point(x_address_end * imgwidth, y_address_end * imgheight));
		Mat m_addr = new Mat(matc, rt_addr);

		Rect rt_id = new Rect(new Point(x_id_start * imgwidth, y_id_start * imgheight),
				new Point(x_id_end * imgwidth, y_id_end * imgheight));
		Mat m_id = new Mat(matc, rt_id);
		// Mat m_id = new Mat(this.orimage, rt_id);

		File imageFile = new File("eurotext.tif");
		ITesseract instance = new Tesseract(); // JNA Interface Mapping
		// ITesseract instance = new Tesseract1(); // JNA Direct Mapping
		// File tessDataFolder = LoadLibs.extractTessResources("tessdata"); // Maven
		// build bundles English data
		File tessDataFolder = new File("F:\\Program Files (x86)\\Tesseract-OCR\\tessdata");
		instance.setDatapath(tessDataFolder.getPath());
		instance.setLanguage("chi_sim+num");

		try {

			// String result = instance.doOCR(imageFile);

			int delval = 90; // hs 去背参数

			if (imgwidth < 1000) {
				// 放大两倍
				m_name = pryUp(m_name, 2);

				Imgproc.blur(m_name, m_name, new Size(1, 1));
			}

			delval = avLightForjiashi(m_name);
			m_name = getSimpleImgForJashiz(m_name, 0, delval);
			m_name = ruihua(m_name);

			// 腐蚀/膨胀
			Mat m_dilate = new Mat();
			Mat structElement1 = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(4, 4), new Point(-1, -1));
			Imgproc.dilate(m_name, m_dilate, structElement1);
			// imshow("膨胀", out1);imwrite("膨胀.jpg", out1);

			// 腐蚀,去除离散点
			Mat m_erode = new Mat();
			Imgproc.erode(m_name, m_name, structElement1);
			Imgcodecs.imwrite(f.getAbsolutePath() + "\\" + "name_dilate.jpg", m_dilate);
			Imgcodecs.imwrite(f.getAbsolutePath() + "\\" + "name_erode.jpg", m_name);

			BufferedImage bimage_name = (BufferedImage) HighGui.toBufferedImage(m_name);
			String name = instance.doOCR(bimage_name);
			boolean isok = Imgcodecs.imwrite(f.getAbsolutePath() + "\\" + "name.jpg", m_name);

			if (imgwidth < 1000) {
				// 放大两倍
				m_addr = pryUp(m_addr, 2);
				// m_addr = pryUp(m_addr,2);
			}

			m_addr = pryUp(m_addr, 2);

			delval = avLightForjiashi(m_addr);
			delval -= 1;
			// delval=127;
			System.out.println("delval:" + delval);
			m_addr = getSimpleImgForJashiz(m_addr, 0, delval);
			Imgproc.blur(m_addr, m_addr, new Size(1, 1));
			m_addr = ruihua(m_addr);

			Imgproc.pyrDown(m_addr, m_addr);

			BufferedImage bimage_addr = (BufferedImage) HighGui.toBufferedImage(m_addr);
			String addr = instance.doOCR(bimage_addr);
			isok = Imgcodecs.imwrite(f.getAbsolutePath() + "\\" + "addr" + ".jpg", m_addr);

			if (imgwidth < 1000) {
				// 放大两倍
				m_id = pryUp(m_id, 2);
				m_id = pryUp(m_id, 2);
				// m_id = ruihua(m_id);
				Imgproc.blur(m_id, m_id, new Size(1, 1));
				// Imgproc.pyrDown(m_addr, m_addr);
			}

			isok = Imgcodecs.imwrite(f.getAbsolutePath() + "\\" + "id_o" + ".jpg", m_id);
			delval = avLightForjiashi(m_id);
			// delval=172;
			m_id = getSimpleImgForJashiz(m_id, 0, delval);
			isok = Imgcodecs.imwrite(f.getAbsolutePath() + "\\" + "id_hls" + ".jpg", m_id);
			// m_id=blur(m_id);

			m_id = ruihua(m_id);
			isok = Imgcodecs.imwrite(f.getAbsolutePath() + "\\" + "id_ruihua" + ".jpg", m_id);
			// 针对驾驶证号码数字 特别训练num识别字体
			instance.setLanguage("num");
			BufferedImage bimage_id = (BufferedImage) HighGui.toBufferedImage(m_id);
			String id = instance.doOCR(bimage_id);
			isok = Imgcodecs.imwrite(f.getAbsolutePath() + "\\" + "id" + ".jpg", m_id);

			String output = "name:" + name + "\r\n";
			output += "addr:" + addr + "\r\n";
			output += "id:" + id + "\r\n";
			System.out.println(output);

			transformedImage.setVisible(false);
			txtinfo.setText(output);

		} catch (Exception e) {
			System.err.println(e.getMessage());
		}

		// boolean isok = Imgcodecs.imwrite(f.getAbsolutePath() + "\\" + i + ".jpg", m);

	}

	/**
	 * 网约车证识别
	 * 
	 * @author zj
	 * @date 2018年8月3日
	 */
	@FXML
	private void ocrWycheTest() {
		Mat src2 = new Mat();
		this.image.copyTo(src2, this.image);
		// Mat blurredImage = new Mat();
		// Imgproc.blur(src2, blurredImage, new Size(1, 1));
		Mat matc = new Mat();
		Imgproc.cvtColor(src2, matc, Imgproc.COLOR_GRAY2BGR);

		String path = FourierController.class.getResource("/").getPath();
		File f = new File(path + "\\out");
		if (!f.exists())
			f.mkdirs();

		int imgwidth = matc.width();
		int imgheight = matc.height();

		double xstartrate = 0.179; // 其他字段开始x比例

		double xid_startrate = 0.178;// 身份证开始x比例

		double x_name_start = xstartrate;//
		double x_name_end = 0.30;// 姓名结束x
		double y_name_start = 0.0842;// 姓名开始高度
		double y_name_end = 0.114;// 姓名开始高度

		double x_bir_start = xstartrate;//
		double x_bir_end = 0.3046;// 地址结束x
		double y_bir_start = 0.1658;// 地址开始高度
		double y_bir_end = 0.195;// 地址结束高度

		double x_address_start = xstartrate;//
		double x_address_end = 0.45;// 地址结束x
		double y_address_start = 0.2089;// 地址开始高度
		double y_address_end = 0.2822;// 地址结束高度

		double x_id_start = xid_startrate;//
		double x_id_end = 0.45;// 身份证号结束x
		double y_id_start = 0.308;// 身份证号结束x
		double y_id_end = 0.35;// 身份证号结束y

		Rect rt_name = new Rect(new Point(x_name_start * imgwidth, y_name_start * imgheight),
				new Point(x_name_end * imgwidth, y_name_end * imgheight));
		Mat m_name = new Mat(matc, rt_name);

		Rect rt_bir = new Rect(new Point(x_bir_start * imgwidth, y_bir_start * imgheight),
				new Point(x_bir_end * imgwidth, y_bir_end * imgheight));
		Mat m_bir = new Mat(matc, rt_bir);

		Rect rt_addr = new Rect(new Point(x_address_start * imgwidth, y_address_start * imgheight),
				new Point(x_address_end * imgwidth, y_address_end * imgheight));
		Mat m_addr = new Mat(matc, rt_addr);

		Rect rt_id = new Rect(new Point(x_id_start * imgwidth, y_id_start * imgheight),
				new Point(x_id_end * imgwidth, y_id_end * imgheight));
		Mat m_id = new Mat(matc, rt_id);
		// Mat m_id = new Mat(this.orimage, rt_id);

		File imageFile = new File("eurotext.tif");
		ITesseract instance = new Tesseract(); // JNA Interface Mapping
		// ITesseract instance = new Tesseract1(); // JNA Direct Mapping
		// File tessDataFolder = LoadLibs.extractTessResources("tessdata"); // Maven
		// build bundles English data
		File tessDataFolder = new File("F:\\Program Files (x86)\\Tesseract-OCR\\tessdata");
		instance.setDatapath(tessDataFolder.getPath());
		instance.setLanguage("chi_sim+num");

		try {

			// String result = instance.doOCR(imageFile);

			int delval = 90; // hs 去背参数

			/*
			 * if (imgwidth < 1000) { // 放大两倍 m_name = pryUp(m_name,2);
			 * 
			 * //Imgproc.blur(m_name, m_name, new Size(1, 1)); }
			 * 
			 * delval = avLight(m_name); m_name = getSimpleImgForJashiz(m_name, 0, delval);
			 */
			m_name = ruihua(m_name);

			/*
			 * //腐蚀/膨胀 Mat m_dilate=new Mat(); Mat structElement1 =Imgproc.
			 * getStructuringElement(Imgproc.MORPH_RECT, new Size(4,4),new Point(-1,-1));
			 * Imgproc.dilate(m_name, m_dilate,structElement1); //imshow("膨胀",
			 * out1);imwrite("膨胀.jpg", out1);
			 * 
			 * //腐蚀,去除离散点 Mat m_erode=new Mat();
			 * Imgproc.erode(m_name,m_name,structElement1);
			 * Imgcodecs.imwrite(f.getAbsolutePath() + "\\" + "name_dilate.jpg", m_dilate);
			 * Imgcodecs.imwrite(f.getAbsolutePath() + "\\" + "name_erode.jpg", m_name);
			 * 
			 * 
			 */
			BufferedImage bimage_name = (BufferedImage) HighGui.toBufferedImage(m_name);
			String name = instance.doOCR(bimage_name);
			boolean isok = Imgcodecs.imwrite(f.getAbsolutePath() + "\\" + "name.jpg", m_name);

			/*
			 * if (imgwidth < 1000) { // 放大两倍 m_addr = pryUp(m_addr,2); //m_addr =
			 * pryUp(m_addr,2); }
			 * 
			 * m_addr = pryUp(m_addr,2);
			 * 
			 * 
			 * 
			 * delval -=1; //delval=127; System.out.println("delval:"+delval); m_addr =
			 * getSimpleImgForJashiz(m_addr, 0, delval); Imgproc.blur(m_addr, m_addr, new
			 * Size(1, 1));
			 * 
			 * 
			 * Imgproc.pyrDown(m_addr, m_addr);
			 */

			// Imgproc.blur(m_addr, m_addr, new Size(1, 1));
			// delval = avLightForjiashi(m_addr);
			// m_addr = getSimpleImgForJashiz(m_addr, 0, delval);
			// m_addr = ruihua(m_addr);

			BufferedImage bimage_addr = (BufferedImage) HighGui.toBufferedImage(m_addr);
			String addr = instance.doOCR(bimage_addr);
			isok = Imgcodecs.imwrite(f.getAbsolutePath() + "\\" + "addr" + ".jpg", m_addr);

			// 针对驾驶证号码数字 特别训练num识别字体
			instance.setLanguage("num");

			/*
			 * if (imgwidth < 1000) { // 放大两倍 m_id = pryUp(m_id,2); m_id = pryUp(m_id,2);
			 * 
			 * Imgproc.blur(m_id, m_id, new Size(1, 1));
			 * 
			 * } delval = avLightForjiashi(m_id); //delval=172; m_id =
			 * getSimpleImgForJashiz(m_id, 0, delval); m_id = ruihua(m_id);
			 */

			BufferedImage bimage_bir = (BufferedImage) HighGui.toBufferedImage(m_bir);
			String bir = instance.doOCR(bimage_bir);
			isok = Imgcodecs.imwrite(f.getAbsolutePath() + "\\" + "bir.jpg", m_bir);

			/*
			 * if (imgwidth < 1000) { // 放大两倍 m_bir = pryUp(m_bir,2); m_bir =
			 * pryUp(m_bir,2); Imgproc.blur(m_bir, m_bir, new Size(1, 1)); }
			 * 
			 * delval = avLight(m_bir); m_bir = getSimpleImgForJashiz(m_bir, 0, delval);
			 * m_bir = ruihua(m_bir);
			 */
			BufferedImage bimage_id = (BufferedImage) HighGui.toBufferedImage(m_id);
			String id = instance.doOCR(bimage_id);
			isok = Imgcodecs.imwrite(f.getAbsolutePath() + "\\" + "id" + ".jpg", m_id);

			String output = "name:" + name + "\r\n";

			output += "bir:" + bir + "\r\n";
			output += "addr:" + addr + "\r\n";
			output += "id:" + id + "\r\n";
			System.out.println(output);

			transformedImage.setVisible(false);
			txtinfo.setText(output);

		} catch (Exception e) {
			System.err.println(e.getMessage());
		}
	}

	@FXML
	private void hebing() {

		Mat frame = new Mat(this.orimage.rows() * 2, this.orimage.cols() * 2, this.orimage.type());
		Rect maxRect = new Rect(0, 0, this.orimage.rows() * 2, this.orimage.cols() * 2);

		/*
		 * ;
		 * 
		 * Mat f1=new Mat(); Rect roiArea=new Rect(0, this.orimage.rows(), 0,
		 * this.orimage.cols()); this.orimage.copyTo(f1, this.orimage);
		 * 
		 * Mat dstRoi=new Mat(frame,maxRect); Core.add(f1, f1, dstRoi);
		 * 
		 */

		Mat src1 = new Mat();
		Mat src2 = new Mat();
		this.orimage.copyTo(src1, this.orimage);
		this.image.copyTo(src2, this.image);

		Rect roiArea = new Rect(0, 0, src1.width(), src1.height());
		Rect roiArea2 = new Rect(0, src1.height(), src1.width(), src1.height());
		Rect roiArea3 = new Rect(src1.width(), 0, src1.width(), src1.height());
		Rect roiArea4 = new Rect(src1.width(), src1.height(), src1.width(), src1.height());

		Mat src1Roi = new Mat(src1, roiArea);
		Mat dstRoi = new Mat(frame, roiArea);
		// Core.add(src1Roi, src1Roi, dstRoi);
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

	/**
	 * Init the needed variables
	 */
	protected void init() {

		// bind a text property with the string containing the current range of
		// HSV values for object detection
		hsvValuesProp = new SimpleObjectProperty<>();
		this.hsvCurrentValues.textProperty().bind(hsvValuesProp);

		this.fileChooser = new FileChooser();
		this.image = new Mat();
		this.planes = new ArrayList<>();
		this.complexImage = new Mat();

		// face
		this.faceCascade = new CascadeClassifier();
		this.absoluteFaceSize = 0;

		this.faceCascade.load("resources/lbpcascades/lbpcascade_frontalface.xml");

		String path = FourierController.class.getResource("/").getPath();
		File f = new File(path + "\\input\\1.jpg");
		if (!f.exists())
			System.out.println(f.getAbsolutePath() + "不存在！");
		else
			loadImageFile(f);
	}

	protected void loadImageFile(File file) {
		// show the open dialog window

		if (file != null) {

			this.orimage = Imgcodecs.imread(file.getAbsolutePath());

			// read the image in gray scale
			this.image = Imgcodecs.imread(file.getAbsolutePath(), Imgcodecs.CV_LOAD_IMAGE_GRAYSCALE);
			// show the image
			this.updateImageView(originalImage, Utils.mat2Image(this.orimage));
			// set a fixed width
			this.originalImage.setFitWidth(450);
			this.originalImage.setFitHeight(250);
			// preserve image ratio
			this.originalImage.setPreserveRatio(true);
			// update the UI
			this.transformButton.setDisable(false);
			this.doCannyBtn.setDisable(false);

			// empty the image planes and the image views if it is not the first
			// loaded image
			if (!this.planes.isEmpty()) {
				this.planes.clear();
				this.transformedImage.setImage(null);
				// this.antitransformedImage.setImage(null);
			}

		}
	}

	/**
	 * Load an image from disk
	 */
	@FXML
	protected void loadImage() {
		// show the open dialog window
		File file = this.fileChooser.showOpenDialog(this.stage);
		if (file != null) {

			this.orimage = Imgcodecs.imread(file.getAbsolutePath());

			// read the image in gray scale
			this.image = Imgcodecs.imread(file.getAbsolutePath(), Imgcodecs.CV_LOAD_IMAGE_GRAYSCALE);
			// show the image
			this.updateImageView(originalImage, Utils.mat2Image(this.orimage));
			// set a fixed width
			this.originalImage.setFitWidth(550);
			this.originalImage.setFitHeight(350);
			// preserve image ratio
			// this.originalImage.setPreserveRatio(true);
			// update the UI
			// this.transformButton.setDisable(false);
			// this.doCannyBtn.setDisable(false);

			// empty the image planes and the image views if it is not the first
			// loaded image
			if (!this.planes.isEmpty()) {
				this.planes.clear();
				this.transformedImage.setImage(null);
				this.antitransformedImage.setImage(null);
			}

		}
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
		view.setVisible(true);
		Utils.onFXThread(view.imageProperty(), image);
	}

}
