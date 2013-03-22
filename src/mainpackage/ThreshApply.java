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

		@Override
		public void map(Text key, Image value, Context context)
				throws IOException, InterruptedException {

			// Threshold preliminaries
			File threshdir = new File(context.getConfiguration().get(
					"mapreduce.imagerecordreader.threshpath"));
			int threshold = -1;
			
			String width = "", height = "";
			File[] listfiles = threshdir.listFiles();
			for (File threshfile : listfiles) {
				BufferedReader reader = new BufferedReader(new FileReader(
						threshfile));
				String line, thresholdstr = "";
				while ((line = reader.readLine()) != null) {
					if (StringUtils.contains(line, key.toString())) {
						StringTokenizer tok = new StringTokenizer(line);
						tok.nextToken();
						width = tok.nextToken();
						height = tok.nextToken();
						// threshold file format:  <filename> width height threshold
						thresholdstr = tok.nextToken();
						threshold = Integer.valueOf(thresholdstr);
						break;
					}
				}
				if (threshold != -1)
					break;
			}

			IplImage img = value.getImage();
			// IplImage greyscale = cvt
			IplImage imgray = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 1);
			cvCvtColor(img, imgray, CV_RGB2GRAY);
			cvThreshold(imgray, imgray, threshold, 255, CV_THRESH_BINARY);
			
			StringBuilder b = new StringBuilder();
			b.append(key.toString());
			b.append(" " + width);
			b.append(" " + height);
			
			context.write(new Text(b.toString()), new Image(img));
		}
	}

	public static class Reduce extends Reducer<Text, Image, Text, Image> {
		
		private static Logger logger = Logger.getLogger(ThreshApply.Reduce.class.getName());
		
		@Override
		public void reduce(Text key, Iterable<Image> values, Context context)
				throws IOException, InterruptedException {
			StringTokenizer tok = new StringTokenizer(key.toString());
			String filename = tok.nextToken();
			int width = Integer.valueOf(tok.nextToken());
			int height = Integer.valueOf(tok.nextToken());
			
			boolean byPixel = context.getConfiguration().getBoolean("mapreduce.imagerecordreader.windowbypixel", false);
			
			// splits based on configuration parameters
			int totalXSplits = 0;
			int totalYSplits = 0;
			int xSplitPixels = 0;
			int ySplitPixels = 0;
			int sizePercent = 0;
			int sizePixel = 0;
			int borderPixel = 0;
			int heightPart = 0;
			int widthPart = 0;
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
			
			
			Iterator it = values.iterator();
			
			int currentSplit = 0;
			IplImage bigimage = cvCreateImage(new CvSize(width, height), IPL_DEPTH_8U, 1);
			IplImage imagepart;
			WindowInfo window;		
			
			while (it.hasNext()) {
				imagepart = ((Image)it.next()).getImage();
				window  = new WindowInfo();
				int x = currentSplit % totalXSplits;
				int y = currentSplit / totalYSplits;

				// Deal with partial windows
				if (x * xSplitPixels + width > width) {
					widthPart = width - x * xSplitPixels;
				}
				if (y * ySplitPixels + height > height) {
					heightPart = height - y * ySplitPixels;
				}

				window.setParentInfo(x * xSplitPixels, y * ySplitPixels,
						height, width);
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
				cvSetImageROI(bigimage, roi);
				
				logger.log(java.util.logging.Level.INFO, "currentsplit " + currentSplit);
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
