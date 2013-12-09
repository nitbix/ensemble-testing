package helpers;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.sql.Connection;
import java.sql.SQLException;

public interface DBConnect {
	public Connection connect() throws FileNotFoundException, IOException, SQLException;
}
