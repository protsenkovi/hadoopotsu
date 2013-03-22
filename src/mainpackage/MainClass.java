package mainpackage;



import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.util.ToolRunner;
import org.codehaus.jackson.map.util.ArrayBuilders;


public class MainClass {
	public static void main(String[] args) throws Exception {
		String tempdir = "e:/develop/eclipseworkspace/tmp";
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

