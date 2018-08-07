package com.kxjl.opencv;


import java.io.IOException;

import org.opencv.core.Core;


import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.Scene;
import javafx.scene.layout.BorderPane;
import javafx.scene.control.ScrollPane;
import javafx.stage.Stage;

public class FMain extends Application {

	// the main stage
	private Stage primaryStage;

	@Override
	public void start(Stage primaryStage) throws IOException {
		// load the FXML resource
		FXMLLoader loader = new FXMLLoader(getClass().getResource("FormFx.fxml"));
		BorderPane root = (BorderPane) loader.load();
		// set a whitesmoke background
		root.setStyle("-fx-background-color: whitesmoke;");
		Scene scene = new Scene(root, 1400, 800);
		scene.getStylesheets().add(getClass().getResource("application.css").toExternalForm());
		// create the stage with the given title and the previously created
		// scene
		this.primaryStage = primaryStage;
		this.primaryStage.setTitle("OpenCv Test");
		this.primaryStage.setScene(scene);
		this.primaryStage.show();

		// init the controller
		FourierController controller = loader.getController();
		//controller.setStage(this.primaryStage);
		controller.init();
	}

	public static void main(String[] args) {

		// load the native OpenCV library
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		
		launch(args);
	}
}
