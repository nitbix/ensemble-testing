package helpers;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.InputStream;

public class FileLoader {
	public InputStream openOrFind(String filename) throws FileNotFoundException
	{
		File f= new File(filename);
		if(f.exists())
			return new FileInputStream(filename);
		else
			return this.getClass().getResourceAsStream("/" + filename);
	}
}
