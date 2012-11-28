package helpers;

public class Labeler {
	private String problem;
	private String etType;
	private String etf;
	private String mlf;
	private String agg;
	private int dataSetSize;
	
	public Labeler(String problem, String etType, String etf, String mlf, String agg, int dataSetSize) {
		this.problem = problem;
		this.etType = etType;
		this.etf = etf;
		this.mlf = mlf;
		this.agg = agg;
		this.dataSetSize = dataSetSize;
	}
	public String get(int size) {
		return problem + "," +etType + "," + etf + ","
		   + mlf + "," + agg + "," + size + "," + dataSetSize;
		}
}
