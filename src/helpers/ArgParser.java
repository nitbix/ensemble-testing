package helpers;

import java.util.ArrayList;
import java.util.List;

import org.encog.engine.network.activation.ActivationFunction;
import org.encog.engine.network.activation.ActivationReLU;
import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.engine.network.activation.ActivationSoftMax;
import org.encog.engine.network.activation.ActivationTANH;
import org.encog.ensemble.EnsembleAggregator;
import org.encog.ensemble.EnsembleMLMethodFactory;
import org.encog.ensemble.EnsembleTrainFactory;
import org.encog.ensemble.aggregator.Averaging;
import org.encog.ensemble.aggregator.MajorityVoting;
import org.encog.ensemble.aggregator.MetaClassifier;
import org.encog.ensemble.aggregator.WeightedAveraging;
import org.encog.ensemble.ml.mlp.factory.MultiLayerPerceptronFactory;
import org.encog.ensemble.training.BackpropagationFactory;
import org.encog.ensemble.training.LevenbergMarquardtFactory;
import org.encog.ensemble.training.ManhattanPropagationFactory;
import org.encog.ensemble.training.ResilientPropagationFactory;
import org.encog.ensemble.training.ScaledConjugateGradientFactory;
import org.encog.neural.networks.training.propagation.resilient.RPROPType;

import techniques.AdaBoostET;
import techniques.BaggingET;
import techniques.EvaluationTechnique;
import techniques.DropoutET;
import techniques.StackingET;

public class ArgParser {
	
	public static class BadArgument extends Exception {
		/**
		 * Means a bad argument has been passed to an argparser. Null is evil, exceptions are good.
		 */
		private static final long serialVersionUID = 3153481788294315535L;
	}
	
	public enum TrainFactories {
		BACKPROP,
		RPROP,
		SCG,
		MANHATTAN,
		LMA,
		ARPROP
	}
	
	public enum MLMethodFactories {
		MLP,
	}
	
	public enum Activations {
		SIGMOID,
		SOFTMAX,
		RELU,
		TANH
	}
	
	public enum Aggregators {
		MAJORITYVOTING,
		AVERAGING,
		WEIGHTEDAVERAGING,
		METACLASSIFIER,
	}
	
	public enum Techniques {
		BAGGING,
		ADABOOST,
		STACKING,
		DROPOUT,
		CURVES //this is a special case to extract training curves
	}

	public static List<Integer> intList(String string) {
		ArrayList<Integer> res = new ArrayList<Integer>();
		for (String value: string.split(",")) {
			res.add(Integer.parseInt(value));
		}
		return res;
	}

	public static List<Double> doubleList(String string) {
		ArrayList<Double> res = new ArrayList<Double>();
		for (String value: string.split(",")) {
			res.add(Double.parseDouble(value));
		}
		return res;
	}

	public static int intSingle(String string) {
		return Integer.parseInt(string);
	}

	public static double doubleSingle(String string) {
		return Double.parseDouble(string);
	}

	public static EnsembleTrainFactory ETF(String string) throws BadArgument {
		String values[] = string.split("-");
		switch (TrainFactories.valueOf(values[0].toUpperCase())) {
			case BACKPROP: return new BackpropagationFactory();
			case RPROP:
			{
				ResilientPropagationFactory rpf = new ResilientPropagationFactory();
				if(values.length > 1 && !values[1].isEmpty())
				{
					rpf.setDropoutRate(doubleSingle(values[1]));
				}
				return rpf;
			}
			case ARPROP:
			{
				ResilientPropagationFactory rpf = new ResilientPropagationFactory();
				rpf.setRPROPType(RPROPType.ARPROP);
				if(values.length > 1 && !values[1].isEmpty())
				{
					rpf.setDropoutRate(doubleSingle(values[1]));
				}
				return rpf;
			}
			case SCG: return new ScaledConjugateGradientFactory();
			case MANHATTAN:
				ManhattanPropagationFactory mpf = new ManhattanPropagationFactory();
				if(values.length > 1)
					mpf.setLearningRate(doubleSingle(values[1]));
				return mpf;
			case LMA: return new LevenbergMarquardtFactory();
			default: throw new BadArgument();
		}
	}

	public static EnsembleMLMethodFactory MLF(String string) throws BadArgument {
		String values[] = string.split(":");
		switch (MLMethodFactories.valueOf(values[0].toUpperCase()))
		{
			case MLP:
				MultiLayerPerceptronFactory mlp = new MultiLayerPerceptronFactory();
				String activations[] = values[2].split(",");
				List<Double> dropoutRates = null;
				if (values.length == 4)
				{
					dropoutRates = doubleList(values[3]);
				} 
				if(values.length == 3 || values.length == 4)
				{
					if(activations.length == 1)
					{
						mlp.setParameters(intList(values[1]), activation(activations[0]),dropoutRates);
					} else if (activations.length == 2) 
					{
						mlp.setParameters(intList(values[1]), activation(activations[0]), activation(activations[1]),dropoutRates);											
					} else if (activations.length == 3) 
					{
						mlp.setParameters(intList(values[1]), activation(activations[0]), activation(activations[1]), activation(activations[2]),dropoutRates);											
					}
				} else { 
					throw new BadArgument();
				}
				return mlp;
			default: throw new BadArgument();
		}
	}

	public static ArrayList<EnsembleMLMethodFactory> MLFS(String string) throws BadArgument {
		ArrayList<EnsembleMLMethodFactory> res = new ArrayList<EnsembleMLMethodFactory>();
		String values[] = string.split("_");
		for (String value: values) {
			res.add(MLF(value));
		}
		return res;
	}
	private static ActivationFunction activation(String string) throws BadArgument {
		switch (Activations.valueOf(string.toUpperCase())) {
			case SIGMOID: return new ActivationSigmoid();
			case SOFTMAX: return new ActivationSoftMax();
			case RELU: return new ActivationReLU();
			case TANH: return new ActivationTANH();
			default: throw new BadArgument();
		}
	}

	public static EnsembleAggregator AGG(String string) throws BadArgument {
		String values[] = string.split("-");
		switch (Aggregators.valueOf(values[0].toUpperCase())) {
			case AVERAGING: return new Averaging();
			case WEIGHTEDAVERAGING: return new WeightedAveraging(null);
			case MAJORITYVOTING: return new MajorityVoting();
			case METACLASSIFIER:
				boolean adaptive = false;
				if(values.length > 4 && values[4].equals("rprop_adaptive"))
				{
					values[4] = "rprop";
					adaptive = true;
				}
				String etf = values[3];
				for(int i = 4; i < values.length; i++)
				{
					if(values[i].equals("adaptive"))
					{
						adaptive = true;
					}
					else
					{
						etf += "-" + values[i];
					}
				}
				return new MetaClassifier(doubleSingle(values[2]),MLF(values[1]), ETF(etf), adaptive);
			default: throw new BadArgument();
		}
	}

	public static ProblemDescription problem(String string) throws BadArgument {
		try {
			return new ProblemDescriptionLoader(string);
		} catch (ProblemDescriptionLoader.BadArgument e) {
			throw new BadArgument();
		}
	}

	public static EvaluationTechnique technique(String etType, List<Integer> sizes,
			Integer dataSetSize, ChainParams fullLabel, EnsembleMLMethodFactory mlf,
			EnsembleTrainFactory etf, EnsembleAggregator agg, DataLoader dataLoader,
			int maxIterations, int maxLoops) throws BadArgument {
		String values[] = etType.split("-");
		switch (Techniques.valueOf(values[0].toUpperCase())) {
			case BAGGING: return new BaggingET(sizes,dataSetSize,maxIterations,maxLoops,fullLabel,mlf,etf,agg);
			case CURVES: return new BaggingET(sizes,dataSetSize,maxIterations,maxLoops,fullLabel,mlf,etf,agg,false);
			case ADABOOST: return new AdaBoostET(sizes,dataSetSize,maxIterations,maxLoops,fullLabel,mlf,etf,agg);
			case STACKING: return new StackingET(sizes,dataSetSize,maxIterations,maxLoops,fullLabel,mlf,etf,agg);
			case DROPOUT: return new DropoutET(dataSetSize,fullLabel,maxIterations,maxLoops,mlf,etf,agg,doubleSingle(values[1]));
			default: throw new BadArgument();
		}
	}

}
