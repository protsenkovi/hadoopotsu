package mainpackage;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.Iterator;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

import static com.googlecode.javacv.cpp.opencv_core.*;
import static com.googlecode.javacv.cpp.opencv_imgproc.*;

import edu.vt.io.Image;
import edu.vt.io.LongArrayWritable;
import edu.vt.input.ImageInputFormat;

public class Histogram extends Configured implements Tool {
	public static class Map extends
			Mapper<Text, Image, Text, LongArrayWritable> {
		private final static LongWritable one = new LongWritable(1);

		@Override
		public void map(Text key, Image value, Context context)
				throws IOException, InterruptedException {

			// Convert to gray scale image
			IplImage im1 = value.getImage();
			IplImage im2 = cvCreateImage(cvSize(im1.width(), im1.height()),
					IPL_DEPTH_8U, 1);
			cvCvtColor(im1, im2, CV_BGR2GRAY);

			// Initialize histogram array
			LongWritable[] histogram = new LongWritable[256];
			for (int i = 0; i < histogram.length; i++) {
				histogram[i] = new LongWritable();
			}

			// Compute histogram
			ByteBuffer buffer = im2.getByteBuffer();
			while (buffer.hasRemaining()) {
				int val = buffer.get() + 128;
				histogram[val].set(histogram[val].get() + 1);
			}

			context.write(key, new LongArrayWritable(histogram));
		}
	}

	public static class Combine extends
			Reducer<Text, LongArrayWritable, Text, LongArrayWritable> {

		@Override
		public void reduce(Text key, Iterable<LongArrayWritable> values,
				Context context) throws IOException, InterruptedException {

			// Initialize histogram array
			LongWritable[] histogram = new LongWritable[256];
			for (int i = 0; i < histogram.length; i++) {
				histogram[i] = new LongWritable();
			}

			// Sum the parts
			Iterator<LongArrayWritable> it = values.iterator();
			while (it.hasNext()) {
				LongWritable[] part = (LongWritable[]) it.next().toArray();
				for (int i = 0; i < histogram.length; i++) {
					histogram[i].set(histogram[i].get() + part[i].get());
				}
			}

			context.write(key, new LongArrayWritable(histogram));
		}
	}

	public static class Reduce extends
			Reducer<Text, LongArrayWritable, Text, Text> {

		@Override
		public void reduce(Text key, Iterable<LongArrayWritable> values,
				Context context) throws IOException, InterruptedException {

			// Initialize histogram array
			LongWritable[] histogram = new LongWritable[256];
			for (int i = 0; i < histogram.length; i++) {
				histogram[i] = new LongWritable();
			}

			// Sum the parts
			Iterator<LongArrayWritable> it = values.iterator();
			while (it.hasNext()) {
				LongWritable[] part = (LongWritable[]) it.next().toArray();
				for (int i = 0; i < histogram.length; i++) {
					histogram[i].set(histogram[i].get() + part[i].get());
				}
			}

			// �����

			// ������ ��� ��������������� �����:
			int m = 0; // m - ����� ����� ���� �����, ����������� �� ���������
						// �� ��������
			int n = 0; // n - ����� ����� ���� �����
			for (int t = 0; t < 256; t++) {
				m += t * histogram[t].get();
				n += histogram[t].get();
			}

			float maxSigma = -1; // ������������ �������� ������������ ���������
			int threshold = 0; // �����, ��������������� maxSigma

			int alpha1 = 0; // ����� ����� ���� ����� ��� ������ 1
			int beta1 = 0; // ����� ����� ���� ����� ��� ������ 1, �����������
							// �� ��������� �� ��������

			// ���������� alpha2 �� �����, �.�. ��� ����� m - alpha1
			// ���������� beta2 �� �����, �.�. ��� ����� n - alpha1

			// t ����������� �� ���� ��������� ��������� ������
			for (int t = 0; t < 256; t++) {
				alpha1 += t * histogram[t].get();
				beta1 += histogram[t].get();

				// ������� ����������� ������ 1.
				float w1 = (float) beta1 / n;
				// �������� ����������, ��� w2 ���� �� �����, �.�. ��� ����� 1 -
				// w1

				// a = a1 - a2, ��� a1, a2 - ������� �������������� ��� �������
				// 1 � 2
				float a = (float) alpha1 / beta1 - (float) (m - alpha1)
						/ (n - beta1);

				// �������, ������� sigma
				float sigma = w1 * (1 - w1) * a * a;

				// ���� sigma ������ ������� ������������, �� ��������� maxSigma
				// � �����
				if (sigma > maxSigma) {
					maxSigma = sigma;
					threshold = t;
				}
			}

			context.write(key, new Text(String.valueOf(threshold)));
		}
	}

	public int run(String[] args) throws Exception {
		// Set various configuration settings
		Configuration conf = getConf();
		conf.setInt("mapreduce.imagerecordreader.windowsizepercent", 23);
		conf.setInt("mapreduce.imagerecordreader.windowoverlappercent", 0);

		// Create job
		Job job = new Job(conf);

		// Specify various job-specific parameters
		job.setJarByClass(Histogram.class);
		job.setJobName("Histogram");

		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(Text.class);
		job.setMapOutputKeyClass(Text.class);
		job.setMapOutputValueClass(LongArrayWritable.class);

		job.setMapperClass(Map.class);
		job.setCombinerClass(Combine.class);
		job.setReducerClass(Reduce.class);

		job.setInputFormatClass(ImageInputFormat.class);
		job.setOutputFormatClass(TextOutputFormat.class);

		// Set input and output paths
		FileInputFormat.addInputPath(job, new Path(args[0]));
		FileOutputFormat.setOutputPath(job, new Path(args[1]));

		return job.waitForCompletion(true) ? 0 : 1;
	}
}