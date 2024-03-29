package mainpackage;



import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.util.ToolRunner;
import org.codehaus.jackson.map.util.ArrayBuilders;

// Program consists of two phases. Calculating histogram and applying threshold.
//
// First phase 
// is run over images. Method getSplits works that split=image, files are not splittable.
// ImageRecordReader divides source image by rectangular subwindows, so this image part is a record.
//
// Map calculates partial histograms.
// Map output: 
//  key format - <filename>
//  value - list of histograms of image parts.
// 
// Reduce collects combined pairs of key defining source image file and list of histograms of all it's parts.
// Each reduce method calculates threshold according to result histogram. This threshold would be written to tempdir path
// in following format:
//     <filename> <threshold> \newline 
//
//
// Second phase
// similarly is run over the same images. except they read as grayscale images(configuration parameter isColor == 0). 
// Directory with files containing calculated thresholds is passed to algorithm
// via Configuration: conf.setStrings("mapreduce.imagerecordreader.threshpath", args[1]); , where args[1] is a tempdir, 
// output path for first map reduce job.
// Division on parts remains unchanged.
// 
// Map finds a threshold for given image filename in all files were with thresholds and applies threshold.
//
// Reduce collects parts and for each key stitch all parts together.
public class MainClass {
	// main args: <input path> <output path>
	public static void main(String[] args) throws Exception {
		String tempdir = args[0].substring(0, args[0].length()-1);
		tempdir = tempdir.substring(0, tempdir.lastIndexOf('/')) + "/tmp";
		String[] newargs = new String[3];
		newargs[0] = args[0];
		newargs[1] = tempdir;
		newargs[2] = args[1];
		args[1] = tempdir;
		int res = ToolRunner.run(new Configuration(), new Histogram(), args);
		int res2 = ToolRunner.run(new Configuration(), new ThreshApply(), newargs); 
		System.exit(res);
	}
}

