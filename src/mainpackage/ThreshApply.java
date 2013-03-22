package mainpackage;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.Iterator;
import java.util.StringTokenizer;

import org.apache.commons.lang.StringUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.Reducer.Context;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import java.util.logging.Logger;
import java.util.logging.Level;

import com.googlecode.javacv.cpp.opencv_core.CvRect;
import com.googlecode.javacv.cpp.opencv_core.IplImage;

import static com.googlecode.javacv.cpp.opencv_core.*;
import static com.googlecode.javacv.cpp.opencv_imgproc.*;

import edu.vt.io.Image;
import edu.vt.io.LongArrayWritable;
import edu.vt.io.WindowInfo;
import edu.vt.input.ImageInputFormat;
import edu.vt.output.ImageOutputFormat;

public class ThreshApply extends Configured implements Tool {
	public static class Map extends Mapper<Text, Image, Text, Image> {
		
		private final static LongWritable one = new LongWritable(1);
		private static Logger logger = Logger.getLogger(ThreshApply.Map.class.getName());
		private boolean debug = false;
		@Override
		public void map(Text key, Image value, Context context)
				throws IOException, InterruptedException {
			debug = context.getConfiguration().getBoolean("mapreduce.debug", true);
			
			if (debug)
				logger.log(java.util.logging.Level.INFO, "VLPR *****************MAP " + key.toString());
			
			// Threshold preliminaries
			File threshdir = new File(context.getConfiguration().get("mapreduce.imagerecordreader.threshpath"));
			int threshold = -1;			
			String width = "", height = "";
			
			// Scan directory with calculated thresholds. Look through files and try to find threshold
			// corresponding to source image file name. 
			File[] listfiles = threshdir.listFiles();
			for (File threshfile : listfiles) {
				BufferedReader reader = new BufferedReader(new FileReader(
						threshfile));
				String line, thresholdstr = "";
				while ((line = reader.readLine()) != null) {
					if (StringUtils.contains(line, key.toString())) { 
						// threshold file format:  <filename> width height threshold
						StringTokenizer tok = new StringTokenizer(line);
						tok.nextToken();
						width = tok.nextToken();
						height = tok.nextToken();
						thresholdstr = tok.nextToken();
						threshold = Integer.valueOf(thresholdstr);
						break;
					}
				}
				if (threshold != -1)
					break;
			}
			
			if (debug)
				logger.log(java.util.logging.Level.INFO, "width "+ width + " height " + height);
			
			// Getting image piece. In case threshold is not found for some reasons return unchanged piece.
			// Not sure it is a right behavior, it's exception and produces unnecessary computations.
			IplImage img = value.getImage();; 
			IplImage imgray;
			if (threshold != -1) {
				imgray = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 1);
				cvCvtColor(img, imgray, CV_RGB2GRAY);
				cvThreshold(imgray, imgray, threshold, 255, CV_THRESH_BINARY);
				
				StringBuilder b = new StringBuilder();
				b.append(key.toString());
				b.append(" " + width);
				b.append(" " + height);
				context.write(new Text(b.toString()), new Image(imgray));
			} else {
				StringBuilder b = new StringBuilder();
				b.append(key.toString());
				b.append(" " + width);
				b.append(" " + height);
				context.write(new Text(b.toString()), new Image(img));
			}		
		}
	}

	public static class Reduce extends Reducer<Text, Image, Text, Image> {
		
		private static Logger logger = Logger.getLogger(ThreshApply.Reduce.class.getName());
		private boolean debug = false;
		private int currentSplit;
		
		@Override
		public void reduce(Text key, Iterable<Image> values, Context context)
				throws IOException, InterruptedException {
			debug = context.getConfiguration().getBoolean("mapreduce.debug", true);
			
			if (debug)
				logger.log(java.util.logging.Level.INFO, "VLPR *********************************REDUCE key: " + key.toString());
			
			// Retrieve filename, width, height of source image from key.
			StringTokenizer tok = new StringTokenizer(key.toString());
			String filename = tok.nextToken();
			int width = Integer.valueOf(tok.nextToken());
			int height = Integer.valueOf(tok.nextToken());			
			
			// Splits based on configuration parameters
			int totalXSplits = 0;
			int totalYSplits = 0;
			int xSplitPixels = 0;
			int ySplitPixels = 0;
			int sizePercent = 0;
			int sizePixel = 0;
			int borderPixel = 0;
			currentSplit = 0;			
			boolean byPixel = context.getConfiguration().getBoolean("mapreduce.imagerecordreader.windowbypixel", false);
			
			// Ensure that value is not negative
			borderPixel = context.getConfiguration().getInt("mapreduce.imagerecordreader.borderPixel", 0);
			if (borderPixel < 0) {
				borderPixel = 0;
			}

			// Ensure that percentage is between 0 and 100
			sizePercent = context.getConfiguration().getInt(
					"mapreduce.imagerecordreader.windowsizepercent", 100);
			if (sizePercent < 0 || sizePercent > 100) {
				sizePercent = 100;
			}

			// Ensure that value is not negative
			sizePixel = context.getConfiguration().getInt("mapreduce.imagerecordreader.windowsizepixel",
					Integer.MAX_VALUE);
			if (sizePixel < 0) {
				sizePixel = 0;
			}
			
			if (byPixel) {
				xSplitPixels = sizePixel;
				ySplitPixels = sizePixel;
				totalXSplits = (int) Math.ceil(width / Math.min(xSplitPixels, width));
				totalYSplits = (int) Math.ceil(height / Math.min(ySplitPixels, height));
			} else {
				xSplitPixels = (int) (width * (sizePercent / 100.0));
				ySplitPixels = (int) (height * (sizePercent / 100.0));
				totalXSplits = (int) Math.ceil(width / (double) Math.min(xSplitPixels, width));
				totalYSplits = (int) Math.ceil(height / (double) Math.min(ySplitPixels, height));
			}
					
			IplImage bigimage = cvCreateImage(new CvSize(width, height), IPL_DEPTH_8U, 1);
			IplImage imagepart;
			WindowInfo window;		
			
			Iterator it = values.iterator();
			while (it.hasNext()) {
				imagepart = ((Image)it.next()).getImage();
				window  = new WindowInfo();
				int widthPart = xSplitPixels;
				int heightPart = ySplitPixels;
				int x = currentSplit % totalXSplits;
				int y = currentSplit / totalYSplits;

				// Deal with partial windows
				if (x * xSplitPixels + widthPart > width) {
					widthPart = width - x * xSplitPixels;
				}
				if (y * ySplitPixels + heightPart > height) {
					heightPart = height - y * ySplitPixels;
				}

				window.setParentInfo(x * xSplitPixels, y * ySplitPixels, height, width);
				window.setWindowSize(heightPart, widthPart);
								
				// Calculate borders
				int top = 0;
				int bottom = 0;
				int left = 0;
				int right = 0;

				if (window.getParentXOffset() > borderPixel) {
					left = borderPixel;
				}
				if (window.getParentYOffset() > borderPixel) {
					top = borderPixel;
				}
				if (window.getParentXOffset() + borderPixel + window.getWidth() < window
						.getParentWidth()) {
					right = borderPixel;
				}
				if (window.getParentYOffset() + borderPixel
						+ window.getHeight() < window.getParentHeight()) {
					bottom = borderPixel;
				}
								
				
				window.setBorder(top, bottom, left, right);				
				CvRect roi = window.computeROI();
				
				if (debug) {
					logger.log(java.util.logging.Level.INFO, "VLPR currentsplit " + currentSplit + " wPart: " + widthPart + " hPart: " + heightPart);
					logger.log(java.util.logging.Level.INFO, "VLPR imagechannels " + imagepart.nChannels() + " imagedepth " + imagepart.depth());				
					logger.log(java.util.logging.Level.INFO, "VLPR x: " + x + " y: " + y + " xSplitPixels: " + xSplitPixels + " ySplitPixels: " + ySplitPixels);
					logger.log(java.util.logging.Level.INFO, "VLPR width  " + width);
					logger.log(java.util.logging.Level.INFO, "VLPR height " + height);
					logger.log(java.util.logging.Level.INFO, "VLPR sizePercent " + sizePercent);
					logger.log(java.util.logging.Level.INFO, "VLPR roi  w: " + roi.width() + " h: " + roi.height() + " x: " + roi.x() + " y:" + roi.y());
					logger.log(java.util.logging.Level.INFO, "VLPR border top: " + top + " bottom: " + bottom + " left: " + left + " right:" + right);
				}
				
				cvSetImageROI(bigimage, roi);				
				
				// copy sub-image
				cvCopy(imagepart, bigimage, null);
				cvResetImageROI(bigimage); 
				
				
				currentSplit++;
			}
			context.write(new Text(filename), new Image(bigimage));
		}
	}

	public int run(String[] args) throws Exception {
		// Set various configuration settings
		Configuration conf = getConf();
		conf.setInt("mapreduce.imagerecordreader.windowsizepercent", 23);
		conf.setInt("mapreduce.imagerecordreader.windowoverlappercent", 0);
		conf.setStrings("mapreduce.imagerecordreader.threshpath", args[1]);
		conf.setBoolean("mapreduce.debug", true);

		// Create job
		Job job = new Job(conf);

		// Specify various job-specific parameters
		job.setJarByClass(ThreshApply.class);
		job.setJobName("ThreshApply");

		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(Image.class);

		job.setMapperClass(Map.class);
		job.setReducerClass(Reduce.class);
		// job.setNumReduceTasks(0);

		job.setInputFormatClass(ImageInputFormat.class);
		job.setOutputFormatClass(ImageOutputFormat.class);

		// Set input and output paths
		FileInputFormat.addInputPath(job, new Path(args[0]));
		FileOutputFormat.setOutputPath(job, new Path(args[2]));

		int ok = job.waitForCompletion(true) ? 0 : 1;
		// Optional
		Path tmppath = new Path(args[1]);
		tmppath.getFileSystem(conf).delete(tmppath, true);
		return ok;
	}
}
