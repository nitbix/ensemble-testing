package helpers;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.util.zip.GZIPInputStream;

public class FileLoader {
	@SuppressWarnings("resource") //we expect the outside world to close the stream
	public InputStream openOrFind(String filename) throws FileNotFoundException, IOException
	{
		InputStream fileStream;
		File f= new File(filename);
		if(f.exists())
		{
			fileStream = new FileInputStream(filename);
		}
		else
		{
			InputStream is = this.getClass().getResourceAsStream("/" + filename);
			if(is != null)
			{
				fileStream =  is;
			}
			else
			{
				is = this.getClass().getResourceAsStream(filename);
				if(is != null)
				{
					fileStream = is;
				}
				throw new FileNotFoundException("could not find " + filename);
			}
		}
		if(filename.endsWith(".gz"))
		{
			return new GZIPInputStream(fileStream);
		}
		return fileStream;
	}
}
