package techniques;

import java.util.ArrayList;

import org.encog.ensemble.EnsembleAggregator;
import org.encog.ensemble.EnsembleMLMethodFactory;
import org.encog.ensemble.EnsembleTrainFactory;
import org.encog.ensemble.bagging.Bagging;

import helpers.ChainParams;
import helpers.DataLoader;

public class SingleET extends EvaluationTechnique {

	private int dataSetSize;

	public SingleET(int dataSetSize, ChainParams fullLabel, EnsembleMLMethodFactory mlMethod, EnsembleTrainFactory trainFactory, EnsembleAggregator aggregator) {
		this.dataSetSize = dataSetSize;
		this.label = fullLabel;
		this.mlMethod = mlMethod;
		this.trainFactory = trainFactory;
		this.aggregator = aggregator;
		this.sizes = new ArrayList<Integer>();
		this.sizes.add(1);
	}

	@Override
	public void step(boolean verbose) {
		if(this.hasStepsLeft) {
			ensemble.trainMember(0,trainToError, selectionError, selectionSet, verbose);
		}
		this.hasStepsLeft = false;
	}

	@Override
	public void init(DataLoader dataLoader, int fold) {
		ensemble = new Bagging(1,dataSetSize,mlMethod,trainFactory,aggregator);
		setTrainingSet(dataLoader.getTrainingSet(fold));
		setSelectionSet(dataLoader.getTestSet(fold));
		ensemble.setTrainingData(trainingSet);
	}

}
