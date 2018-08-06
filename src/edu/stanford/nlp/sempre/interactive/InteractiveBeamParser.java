package edu.stanford.nlp.sempre.interactive;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.Comparator;

import com.beust.jcommander.internal.Lists;
import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;

import edu.stanford.nlp.sempre.ChartParserState;
import edu.stanford.nlp.sempre.Derivation;
import edu.stanford.nlp.sempre.DerivationStream;
import edu.stanford.nlp.sempre.Example;
import edu.stanford.nlp.sempre.Formula;
import edu.stanford.nlp.sempre.Json;
import edu.stanford.nlp.sempre.Params;
import edu.stanford.nlp.sempre.Parser;
import edu.stanford.nlp.sempre.ParserState;
import edu.stanford.nlp.sempre.Rule;
import edu.stanford.nlp.sempre.SemanticFn;
import edu.stanford.nlp.sempre.Trie;
import fig.basic.IOUtils;
import fig.basic.IntRef;
import fig.basic.LogInfo;
import fig.basic.Option;
import fig.basic.SetUtils;
import fig.basic.StopWatch;
import fig.basic.StopWatchSet;
import fig.exec.Execution;
import org.apache.commons.lang3.Range;

/**
 * A modified version of the BeamParser, with consideration for use in the interactive setting
 * 
 * @author Percy Liang, sidaw
 */


public class InteractiveBeamParser extends Parser {
	
 
	
  public static class Options {
    @Option
    public int maxNewTreesPerSpan = Integer.MAX_VALUE;
    @Option
    public FloatStrategy floatStrategy = FloatStrategy.Never;
    @Option(gloss = "track these categories")
    public List<String> trackedCats;
    @Option(gloss="Similarity threshold for a rule to be applicable to a non-parsable utterance")
    public double simMin = 0.5;

  }

  public enum FloatStrategy {
    Always, Never, NoParse
  };

  public static Options opts = new Options();
  
  
  Trie trie; // For non-cat-unary rules
  // so that duplicated rules are never added
  Set<Rule> allRules;
  List<Rule> interactiveCatUnaryRules;
  public InteractiveBeamParser(Spec spec) {
    super(spec);
    if (opts.trackedCats != null) {
      opts.trackedCats = opts.trackedCats.stream().map(s -> "$" + s).collect(Collectors.toList());
      LogInfo.logs("Mapped trackedCats to: %s", opts.trackedCats);
    }
    interactiveCatUnaryRules = new LinkedList<>(super.catUnaryRules);
    allRules = new LinkedHashSet<>(super.catUnaryRules);
    // Index the non-cat-unary rules
    trie = new Trie();
    for (Rule rule : grammar.getRules()) {
      addRule(rule);
    }
    if (Parser.opts.visualizeChartFilling)
      this.chartFillOut = IOUtils.openOutAppendEasy(Execution.getFile("chartfill"));
  }

  @Override
  public synchronized void addRule(Rule rule) {
    if (allRules.contains(rule))
      return;
    
    allRules.add(rule);

    if (!rule.isCatUnary()) {
      trie.add(rule);
    } else {
      interactiveCatUnaryRules.add(rule);
    }
  }
  
  /** Deletes rule from parser, depending on if the rule is cat-unary or not
   * @author Akshal Aniche
   * @param rule
   */
  @Override
  public synchronized void removeRule(Rule rule){
	 
	  if (!allRules.contains(rule)) {
	  	return;
	  }

	  List<Rule> list = Collections.singletonList(rule);
	  allRules.removeAll(list);

	  if (!rule.isCatUnary()) {
		  trie.remove(rule);
	  } else {
		  interactiveCatUnaryRules.removeAll(list);
	  }
  }
  
  @Override
  public List<Rule> getCatUnaryRules() {
    return interactiveCatUnaryRules;
  }
  
  
  
  // for grammar induction, just need the formula, do not execute
  public InteractiveBeamParserState parseWithoutExecuting(Params params, Example ex, boolean computeExpectedCounts) {
	  // Parse
    StopWatch watch = new StopWatch();
    watch.start();
    InteractiveBeamParserState state = new InteractiveBeamParserState(this, params, ex);
    state.infer();
    watch.stop();
    state.parseTime = watch.getCurrTimeLong();

    ex.predDerivations = state.predDerivations;
    Derivation.sortByScore(ex.predDerivations);
    // Clean up temporary state used during parsing
    return state;
  }


  @Override
  public ParserState newParserState(Params params, Example ex, boolean computeExpectedCounts) {
    InteractiveBeamParserState coarseState = null;
    if (Parser.opts.coarsePrune) {
    	if (Parser.opts.verbose > 1) {
    		LogInfo.begin_track("Parser.coarsePrune");
    	}
      
      // in this state only the phrases are assigned (no other categories)
      coarseState = new InteractiveBeamParserState(this, params, ex, computeExpectedCounts,
          InteractiveBeamParserState.Mode.bool, null);
     // coarseState.visualizeChart();
      
      coarseState.infer();
      coarseState.keepTopDownReachable();
      if (Parser.opts.verbose > 1) {
    	  LogInfo.end_track();
      }
    }
    return new InteractiveBeamParserState(this, params, ex, computeExpectedCounts, InteractiveBeamParserState.Mode.full,
        coarseState);
  }
}