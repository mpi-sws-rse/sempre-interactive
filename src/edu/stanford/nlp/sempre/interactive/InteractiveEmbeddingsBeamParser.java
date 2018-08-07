package edu.stanford.nlp.sempre.interactive;

import java.util.List;

import edu.stanford.nlp.sempre.Example;
import edu.stanford.nlp.sempre.Params;
import edu.stanford.nlp.sempre.Parser;
import edu.stanford.nlp.sempre.ParserState;
import edu.stanford.nlp.sempre.interactive.embeddings.*;

import fig.basic.Option;
import fig.basic.LogInfo;

import edu.cmu.lti.lexical_db.ILexicalDatabase;
import edu.cmu.lti.lexical_db.NictWordNet;
import edu.cmu.lti.ws4j.RelatednessCalculator;
import edu.cmu.lti.ws4j.impl.HirstStOnge;
import edu.cmu.lti.ws4j.impl.JiangConrath;
import edu.cmu.lti.ws4j.impl.LeacockChodorow;
import edu.cmu.lti.ws4j.impl.Lesk;
import edu.cmu.lti.ws4j.impl.Lin;
import edu.cmu.lti.ws4j.impl.Path;
import edu.cmu.lti.ws4j.impl.Resnik;
import edu.cmu.lti.ws4j.impl.WuPalmer;
import edu.cmu.lti.ws4j.util.WS4JConfiguration;

public class InteractiveEmbeddingsBeamParser extends InteractiveBeamParser {

	public enum SimilarityMeasure {
		LIN, WUP, HSO, PATH, W2V
	}
	
	public static class Options {
		@Option(gloss="What similarity measure to use")
		public SimilarityMeasure sim = SimilarityMeasure.LIN;
		@Option(gloss="Path to word vector embeddings database files")
		public String embeddingsPath="";
	}
	
	public static Options opts = new Options();
	private static ILexicalDatabase db = new NictWordNet();	
	//option of similarity metrics that use 
	public final static RelatednessCalculator[] rcs = 
		{ new Lin(db), new WuPalmer(db), new HirstStOnge(db), new Path(db) };
	
	public final Embeddings embeddings;

	public InteractiveEmbeddingsBeamParser(Spec spec) {
		super(spec);
		if (opts.sim == SimilarityMeasure.W2V) {
			if (opts.embeddingsPath.equals("")) 
				throw new IllegalArgumentException("You need to provide a path to the word vector embeddings database as an option.");
			embeddings = new Embeddings(opts.embeddingsPath);
		}
		else {
			embeddings = null;
		}
	}

	@Override
	public ParserState newParserState(Params params, Example ex, boolean computeExpectedCounts) {
	  InteractiveEmbeddingsBeamParserState coarseState = null;
	    if (Parser.opts.coarsePrune) {
	    	if (Parser.opts.verbose > 1) {
	    		LogInfo.begin_track("Parser.coarsePrune");
	    	}
	      
	      // in this state only the phrases are assigned (no other categories)
	    coarseState = new InteractiveEmbeddingsBeamParserState(this, params, ex, computeExpectedCounts,
	        InteractiveBeamParserState.Mode.bool, null);
	     // coarseState.visualizeChart();
	      
	      coarseState.infer();
	      coarseState.keepTopDownReachable();
	      if (Parser.opts.verbose > 1) {
	    	  LogInfo.end_track();
	      }
	    }
	    return new InteractiveEmbeddingsBeamParserState(this, params, ex, computeExpectedCounts, InteractiveBeamParserState.Mode.full,
	        coarseState);
	  }
}

class InteractiveEmbeddingsBeamParserState extends InteractiveBeamParserState {

	public InteractiveEmbeddingsBeamParserState(InteractiveBeamParser parser,
			Params params, Example ex, boolean computeExpectedCounts,
			Mode mode, InteractiveBeamParserState coarseState) {
		super(parser, params, ex, computeExpectedCounts, mode, coarseState);
	}
	
	public InteractiveEmbeddingsBeamParserState(InteractiveBeamParser parser, Params params, Example ex) {
	    super(parser, params, ex);
	}
	
	
	private double computeSimilarityWordNet(String word1, String word2, int calculator) {
		WS4JConfiguration.getInstance().setMFS(false);
		double s = InteractiveEmbeddingsBeamParser.rcs[calculator].calcRelatednessOfWords(word1, word2);
		//Upper Bound for HSO is 16
		if (calculator == 2)
			s = s / 16;
		return s;
	}
	
	
	private double computeSimilarityWordVector(String word1, String word2){
		Embeddings embeddings = ((InteractiveEmbeddingsBeamParser) parser).embeddings;
		Word w1 = embeddings.getWord(word1);
		Word w2 = embeddings.getWord(word2);
		
		//the word wasn't found in the dictionary
		if (w1.equals(Word.nullWord) || w2.equals(Word.nullWord))
			return 0.0;	
		
		double cosSim = Embeddings.sim(w1, w2); //cosine similarity, range = [-1, 1]
		return (cosSim + 1) / 2.0;
	}
	
	/**
	 * Computes dynamically the length of the LCS between list1 and list2, but augmented by using word similarity when applicable
	 */
	protected double longestCommonSubsequence(List<String> list1, List<String> list2){
		 int len1 = list1.size();
		 int len2 = list2.size();
		 InteractiveEmbeddingsBeamParser.SimilarityMeasure sim = InteractiveEmbeddingsBeamParser.opts.sim ;

		 double[][] subsequence = new double[len1 + 1][len2 + 1];
		  
		 for (int i = 0; i <= len1; i++) 
			 subsequence[i][0] = 0;
		 for (int j = 0; j <= len2; j++) 
			subsequence[0][j] = 0;
		 for (int i = 1; i <= len1; i++) {
			for (int j = 1; j <= len2; j++) {
				String word1 = list1.get(i-1);
				String word2 = list2.get(j-1);
				if (word1.equals(word2)) 
					subsequence[i][j] = 1 + subsequence [i-1][j-1];
				else if (word1.startsWith("$") || word2.startsWith("$"))
					subsequence[i][j] = Math.max(subsequence[i-1][j], subsequence[i][j-1]);
				else {
					double similarity;
					if (sim == InteractiveEmbeddingsBeamParser.SimilarityMeasure.W2V) {
						similarity = computeSimilarityWordVector(word1, word2);
					}
					else {
						similarity = computeSimilarityWordNet(word1, word2, sim.ordinal());
					}
					if (Parser.opts.verbose >= 2)
						LogInfo.logs("%s similarity between %s and %s is %f", sim.toString(), word1, word2, similarity);
					subsequence[i][j] = Math.max(subsequence[i-1][j], Math.max(subsequence[i][j-1], subsequence[i-1][j-1] + similarity));
				}
			}
		}
		
		if (Parser.opts.verbose >=2)
			LogInfo.logs("Subsequence length between %s and %s using %s similarity: %s", list1.toString(), list2.toString(), sim.toString(), subsequence[len1][len2]);
		  
		return subsequence[len1][len2];
	  }
	  
}
