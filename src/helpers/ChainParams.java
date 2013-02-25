package helpers;

public class ChainParams {
	private String problem;
	private String etType;
	private String etf;
	private String mlf;
	private String agg;
	private int dataSetSize;
	
	public ChainParams(String problem, String etType, String etf, String mlf, String agg, int dataSetSize)
	{
		this.problem = problem;
		this.etType = etType;
		this.etf = etf;
		this.mlf = mlf;
		this.agg = agg;
		this.dataSetSize = dataSetSize;
	}
	
	public String getETType()
	{
		return etType;
	}
	
	public String getETF()
	{
		return etf;
	}
	
	public String getMLF()
	{
		return mlf;
	}
	
	public String getAgg()
	{
		return agg;
	}
	
	public String getProblem()
	{
		return problem;
	}
	
	public String get(int size) 
	{
		return problem + "," +etType + "," + etf + "," + mlf + "," + agg + "," + size + "," + dataSetSize;
	}
}
