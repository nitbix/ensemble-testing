package techniques;

import java.util.ArrayList;

import org.encog.ensemble.EnsembleAggregator;
import org.encog.ensemble.EnsembleMLMethodFactory;
import org.encog.ensemble.EnsembleTrainFactory;
import org.encog.ensemble.aggregator.MajorityVoting;
import org.encog.ensemble.stacking.Stacking;

import helpers.ChainParams;
import helpers.DataLoader;

public class DropoutET extends EvaluationTechnique {

	private int dataSetSize;

	public DropoutET(int dataSetSize, ChainParams fullLabel, EnsembleMLMethodFactory mlMethod, EnsembleTrainFactory trainFactory, EnsembleAggregator aggregator, double dropoutRate) {
		this.dataSetSize = dataSetSize;
		this.label = fullLabel;
		this.mlMethod = mlMethod;
		this.trainFactory = trainFactory;
		this.trainFactory.setDropoutRate(dropoutRate);
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
		ensemble = new Stacking(1,dataSetSize,mlMethod,trainFactory,new MajorityVoting());
		setTrainingSet(dataLoader.getTrainingSet(fold));
		setSelectionSet(dataLoader.getTestSet(fold));
		ensemble.setTrainingData(trainingSet);
	}

}
