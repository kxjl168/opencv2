package it.polito.teaching.cv;

import java.awt.AlphaComposite;
import java.awt.Graphics2D;
import java.awt.Image;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.ByteArrayInputStream;
import java.io.InputStream;

import javax.imageio.ImageIO;

import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.highgui.HighGui;

public class KUtil {
	/**
	 * 将Mat类型转化成BufferedImage类型
	 * 
	 * @param amatrix
	 *            Mat对象
	 * @param fileExtension
	 *            文件扩展名
	 * @return
	 */
	public static BufferedImage Mat2Img(Mat mat, String fileExtension) {
		MatOfByte mob = new MatOfByte();

	/*	Image i= HighGui.toBufferedImage(mat);
		i.getGraphics().getb
		HighGui.imencode(fileExtension, mat, mob);*/
		// convert the "matrix of bytes" into a byte array
		byte[] byteArray = mob.toArray();
		BufferedImage bufImage = null;
		try {
			InputStream in = new ByteArrayInputStream(byteArray);
			bufImage = ImageIO.read(in);
		
		} catch (Exception e) {
			e.printStackTrace();
		}
		return bufImage;
	}

	// 将BufferedImage类型转换成Mat类型

	/**
	 * 将BufferedImage类型转换成Mat类型
	 * 
	 * @param bfImg
	 * @param imgType
	 *            bufferedImage的类型 如 BufferedImage.TYPE_3BYTE_BGR
	 * @param matType
	 *            转换成mat的type 如 CvType.CV_8UC3
	 * @return
	 */
	public static Mat Img2Mat(BufferedImage bfImg, int imgType, int matType) {
		BufferedImage original = bfImg;
		int itype = imgType;
		int mtype = matType;

		if (original == null) {
			throw new IllegalArgumentException("original == null");
		}

		if (original.getType() != itype) {
			BufferedImage image = new BufferedImage(original.getWidth(), original.getHeight(), itype);

			Graphics2D g = image.createGraphics();
			try {
				g.setComposite(AlphaComposite.Src);
				g.drawImage(original, 0, 0, null);
			} finally {
				g.dispose();
			}
		}

		byte[] pixels = ((DataBufferByte) original.getRaster().getDataBuffer()).getData();
		Mat mat = Mat.eye(original.getHeight(), original.getWidth(), mtype);
		mat.put(0, 0, pixels);

		return mat;
	}

}
