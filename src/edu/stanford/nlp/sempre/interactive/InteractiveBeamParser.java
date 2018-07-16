package edu.stanford.nlp.sempre.interactive;

import java.util.ArrayList;
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

/**
 * Stores BeamFloatingParser information about parsing a particular example. The
 * actual parsing code lives here.
 *
 * @author Percy Liang
 * @author Roy Frostig
 * @author sidaw
 */
class InteractiveBeamParserState extends ChartParserState {
  public final Mode mode;

  // Modes:
  // 1) Bool: just check if cells (cat, start, end) are reachable (to prune
  // chart)
  // 2) Full: compute everything
  public enum Mode {
    bool, full
  }

  private final InteractiveBeamParser parser;
  private final InteractiveBeamParserState coarseState; // Used to prune
  private final boolean execute;
  
  private boolean extendedParsing = false;
  public List<Derivation> chartList;

  public InteractiveBeamParserState(InteractiveBeamParser parser, Params params, Example ex) {
    super(parser, params, ex, false);
    this.parser = parser;
    this.mode = Mode.full;
    this.coarseState = null;
    this.execute = false;
  }

  public InteractiveBeamParserState(InteractiveBeamParser parser, Params params, Example ex, boolean computeExpectedCounts,
      Mode mode, InteractiveBeamParserState coarseState) {
    super(parser, params, ex, computeExpectedCounts);
    this.parser = parser;
    this.mode = mode;
    this.coarseState = coarseState;
    this.execute = true;
  }
  
  public boolean wasParsingExtended() { return extendedParsing; }

  @Override
  public void infer() {
    if (numTokens == 0)
      return;

    if (parser.verbose(2)){
      LogInfo.begin_track("ParserState.infer");
    }

    // Base case
    for (Derivation deriv : gatherTokenAndPhraseDerivations()) {
      featurizeAndScoreDerivation(deriv);
      addToChart(deriv);
    }
    
    
    if (Parser.opts.verbose >4){
    	LogInfo.logs("before recursive case:");
    	visualizeChart();	
      
    }

    // Recursive case
    for (int len = 1; len <= numTokens; len++)
      for (int i = 0; i + len <= numTokens; i++)
        build(i, i + len);

    if (Parser.opts.verbose > 4){
    	LogInfo.logs("after recursive case");
    	visualizeChart();	
      
    }
    
    if (parser.verbose(2)){
    	LogInfo.end_track();
    }
    	

    // Visualize
    if (parser.chartFillOut != null && Parser.opts.visualizeChartFilling && this.mode != Mode.bool) {
      parser.chartFillOut.println(
          Json.writeValueAsStringHard(new ChartFillingData(ex.id, chartFillingList, ex.utterance, ex.numTokens())));
      parser.chartFillOut.flush();
    }
    
    // putting to predDerivations anything that parses with the rule $ROOT -> something
    setPredDerivations();
    
    if (Parser.opts.verbose > 4){
    	LogInfo.logs("pred derivations = %s", predDerivations);
    }

    for (Derivation deriv : predDerivations) {
    	//this is very weird, I have no idea what is happening here
      deriv.getAnchoredTokens();
      
    }
    this.chartList = this.collectChart();
    
    

    boolean parseFloat = false;
    if (InteractiveBeamParser.opts.floatStrategy == InteractiveBeamParser.FloatStrategy.Always)
      parseFloat = true;
    else if (InteractiveBeamParser.opts.floatStrategy == InteractiveBeamParser.FloatStrategy.NoParse)
      parseFloat = predDerivations.size() == 0;
    else
      parseFloat = false;

    boolean definedLoopVariablesUsedCorrectly = true;
    for (Iterator<Derivation> iter = predDerivations.iterator(); iter.hasNext();) {
      Derivation d = iter.next();
      // checks if there is a loop variable used out of scope ( foreach ... point, foreach ... area)
      if (!SemanticAnalyzer.checkVariables(d)) {
    	  definedLoopVariablesUsedCorrectly = false;
    	  if (Parser.opts.verbose >= 4) {
    		  LogInfo.logs("semantic analyzer fired and removed %s", d);
    	  }
        iter.remove();
      }
    }

    if (mode == Mode.full) {
      // Compute gradient with respect to the predicted derivations
      if (this.execute)
        ensureExecuted();
      if (computeExpectedCounts) {
        expectedCounts = new HashMap<>();
        ParserState.computeExpectedCounts(predDerivations, expectedCounts);
        LogInfo.logs("after calculating expected counts = %s", expectedCounts);
      }
    }
    

    /* If Beam Parser failed to find derivations, try a floating parser */
    if (parseFloat) {
      /*
       * For every base span of the chart, add the derivations from nothing
       * rules
       */
      List<Rule> nothingRules = new ArrayList<Rule>();
      for (Rule rule : parser.grammar.getRules())
        if (rule.isFloating() && rule.rhs.size() == 1 && rule.isRhsTerminals())
          nothingRules.add(rule);
      for (int i = 0; i < numTokens; i++)
        for (Rule rule : nothingRules)
          applyRule(i, i + 1, rule, chart[i][i + 1].get("$TOKEN"));

      /* Traverse the chart bottom up */
      for (int len = 1; len <= numTokens; len++) {
        for (int i = 0; i + len <= numTokens; i++) {
          buildFloating(i, i + len);
        }
      }

      /* Add unique derivations to predDerivations */
      List<Derivation> rootDerivs = chart[0][numTokens].get("$FROOT");
      if (rootDerivs == null)
        rootDerivs = new ArrayList<Derivation>(Derivation.emptyList);

      List<Derivation> actionDerivs = new ArrayList<Derivation>(Derivation.emptyList);
      if (actionDerivs != null) {
        Set<Formula> formulas = new HashSet<Formula>();
        for (Derivation d : rootDerivs) {
          Formula f = d.getFormula();
          if (!formulas.contains(f)) {
            formulas.add(f);
            predDerivations.add(d);
          }
        }
      }
    }
    
    // if we failed to find any (full) derivations
    if (predDerivations.size() == 0) {
    	// we want to print this only in the second iteration, when we were doing full parsing
    	
        if(coarseState != null && Parser.opts.partialParsing == true) {
        	List<String> understandableStrings = eliminateCoveredOnes(this.chartList);
        	LogInfo.logs("PARSER: can't parse the whole utterance. The parts that I can make sense of are: %s", understandableStrings);
        	if (definedLoopVariablesUsedCorrectly == false) {
        		LogInfo.logs("PARSER: Make sure that you're using loop variables correctly. \' foreach point in [some area] ... [use point as a variable name] \' or \' foreach area in [some collection of areas]... [use area as a variable name] \'.");
        	}
        	
//        	for (Derivation d : this.chartList) {
//    			d.printDerivationRecursively();
//    		}
        	
        	if (parser.verbose(2)){
        		LogInfo.begin_track("ParserState.infer trying to extend parsing");
        	}
        	
        	//avoid infinite loop 
        	if (this.secondParsing) {
        		throw new RuntimeException("Trying to extend parsing within a previous extension of parsing.");
        	}
		
        	extendParsing();
        	
        	if (predDerivations.size() != 0) 
        		  this.extendedParsing = true; //signals that extended parsing was used successfully
		        	
        	if (parser.verbose(2)){
        		LogInfo.end_track();
        	}
        }
    }
  }
  
  private LinkedList<String> eliminateCoveredOnes(List<Derivation> chartList){
	  LinkedList<Range> allRanges = new LinkedList<Range>();
	  LinkedList<Range> filteredRanges = new LinkedList<Range>();
	  LinkedList<String> understandableStrings = new LinkedList<String>();
	  for (Derivation d : chartList) {
		  Range derivRange =Range.between(d.start, d.end); 
		  if (!allRanges.contains(derivRange)) {
			  allRanges.add(derivRange);
		  }
	  }
	  
	  for (Range i : allRanges) {
		  boolean notContained = true;
		  for (Range j : allRanges) {
			  if (j != i && j.containsRange(i)) {
				  notContained = false;
				  continue;
			  }
		  }
		  if (notContained == true) {
			  filteredRanges.add(i);
		  }
	  }
	  
	  
	  
//	  LogInfo.logs("all ranges are: %s", allRanges);
//	  LogInfo.logs("filtered ranges are: %s", filteredRanges);
	  
	  filteredRanges.sort( new Comparator<Range>() {
		  @Override
		  public int compare(Range r1, Range r2) {
			  int r1Start = (int)r1.getMinimum();
			  int r2Start = (int)r2.getMinimum();
			  if (r1Start == r2Start){
				  return 0;
			  }
			  else if (r1Start < r2Start) {
				  return -1;
			  }
			  else {
				  return 1;
			  }
		  }
	  });
	  
	  
	  for (Range r : filteredRanges) {
		  understandableStrings.add(ex.phraseString((int)r.getMinimum(), (int)r.getMaximum()));
	  }
	  return understandableStrings;
  }

  private List<Derivation> collectChart() {
	  
    List<Derivation> chartList = Lists.newArrayList();
    for (int len = 1; len <= numTokens; ++len) {
      for (int i = 0; i + len <= numTokens; ++i) {
        for (String cat : chart[i][i + len].keySet()) {
          if (Rule.specialCats.contains(cat))
            continue;
          chartList.addAll(chart[i][i + len].get(cat));
        }
      }
    }
    return chartList;
  }

  // Create all the derivations for the span [start, end).
  protected void build(int start, int end) {
   
	applyNonCatUnaryRules(start, end, start, parser.trie, new ArrayList<Derivation>(), new IntRef(0));
	
    Set<String> cellsPruned = new HashSet<>();
    applyCatUnaryRules(start, end, cellsPruned);
    
    for (Map.Entry<String, List<Derivation>> entry : chart[start][end].entrySet())
      pruneCell(cellsPruned, entry.getKey(), start, end, entry.getValue());
  }

  private static String cellString(String cat, int start, int end) {
    return cat + ":" + start + ":" + end;
  }

  
  // Return number of new derivations added
  private int applyRule(int start, int end, Rule rule, List<Derivation> children) {
    if (Parser.opts.verbose >= 5)
      LogInfo.logs("applyRule %s %s %s %s", start, end, rule, children);
    try {
      if (mode == Mode.full) {
        StopWatchSet.begin(rule.getSemRepn());
        DerivationStream results = rule.sem.call(ex,
            new SemanticFn.CallInfo(rule.lhs, start, end, rule, ImmutableList.copyOf(children)));
        StopWatchSet.end();
        while (results.hasNext()) {
          Derivation newDeriv = results.next();
          featurizeAndScoreDerivation(newDeriv);
          addToChart(newDeriv);
        }
        return results.estimatedSize();
      } else if (mode == Mode.bool) {
        Derivation deriv = new Derivation.Builder().cat(rule.lhs).start(start).end(end).rule(rule)
            .children(ImmutableList.copyOf(children)).formula(Formula.nullFormula).createDerivation();
        addToChart(deriv);
        return 1;
      } else {
        throw new RuntimeException("Invalid mode");
      }
    } catch (Exception e) {
      LogInfo.errors("Composition failed: rule = %s, children = %s", rule, children);
      e.printStackTrace();
      throw new RuntimeException(e);
    }
  }

  // Don't prune the same cell more than once.
  protected void pruneCell(Set<String> cellsPruned, String cat, int start, int end, List<Derivation> derivations) {
    String cell = cellString(cat, start, end);
    if (cellsPruned.contains(cell))
      return;

    cellsPruned.add(cell);
    pruneCell(cell, derivations);
  }

  private boolean canBeRoot(int start, int end) {return start==0 && end==numTokens;};
 
  
  // Apply all unary rules with RHS category.
  // Before applying each unary rule (rule.lhs -> rhsCat), we can prune the cell
  // of rhsCat
  // because we assume acyclicity, so rhsCat's cell will never grow.
  private void applyCatUnaryRules(int start, int end, Set<String> cellsPruned) {
    for (Rule rule : parser.getCatUnaryRules()) {
      if (!coarseAllows(rule.lhs, start, end))
        continue;
      if (rule.lhs.equals(Rule.rootCat) && !canBeRoot(start, end))
        continue;
      String rhsCat = rule.rhs.get(0);
      List<Derivation> derivations = chart[start][end].get(rhsCat);
      if (Parser.opts.verbose >= 5)
        LogInfo.logs("applyCatUnaryRules %s %s %s %s", start, end, rule, chart[start][end]);
      if (derivations == null)
        continue;

      // Prune before applying rules to eliminate cruft!
      pruneCell(cellsPruned, rhsCat, start, end, derivations);

      for (Derivation deriv : derivations) {
        applyRule(start, end, rule, Collections.singletonList(deriv));
      }
    }
  }

  // Strategy: walk along the input on span (start:end) and traverse the trie
  // to get the list of the rules that could apply by matching the RHS.
  // start:end: span we're dealing with.
  // i: current token position
  // node: contains a link to the RHS that could apply.
  // children: the derivations that't we're building up.
  // numNew: Keep track of number of new derivations created
  private void applyNonCatUnaryRules(int start, int end, int i, Trie node, ArrayList<Derivation> children,
      IntRef numNew) {
    if (node == null)
      return;
    if (!coarseAllows(node, start, end))
      return;
    
    if (Parser.opts.verbose >= 5) {
      LogInfo.logs("applyNonCatUnaryRules(start=%d, end=%d, i=%d, children=[%s], %s rules)", start, end, i,
          Joiner.on(", ").join(children), node.rules.size());
    }

    // Base case: our fencepost has walked to the end of the span, so
    // apply the rule on all the children gathered during the walk.
    if (i == end) {
    	
      Iterator<Rule> ruleIterator = node.rules.iterator();
      while (ruleIterator.hasNext()) {
        Rule rule = ruleIterator.next();
        if (coarseAllows(rule.lhs, start, end)) {
        
          numNew.value += applyRule(start, end, rule, children);
        }
      }
      return;
    }

    // Advance terminal token
    applyNonCatUnaryRules(start, end, i + 1, node.next(ex.token(i)), children, numNew);

    // Advance non-terminal category
    for (int j = i + 1; j <= end; j++) {
      for (Map.Entry<String, List<Derivation>> entry : chart[i][j].entrySet()) {
        Trie nextNode = node.next(entry.getKey());
        for (Derivation arg : entry.getValue()) {
          children.add(arg);
          applyNonCatUnaryRules(start, end, j, nextNode, children, numNew);
          children.remove(children.size() - 1);
          if (mode != Mode.full)
            break; // Only need one hypothesis
          if (numNew.value >= InteractiveBeamParser.opts.maxNewTreesPerSpan)
            return;
        }
      }
    }
  }

  /* For each span, apply applicable floating rules */
  protected void buildFloating(int start, int end) {
    for (Rule rule : parser.grammar.getRules()) {
      if (!rule.isFloating() || !coarseAllows(rule.lhs, start, end))
        continue;

      if (rule.rhs.size() == 1) {
        /* Apply cat unary rules simply */
        String rhsCat = rule.rhs.get(0);
        List<Derivation> derivs = chart[start][end].get(rhsCat);

        if (derivs == null)
          continue;

        for (Derivation deriv : derivs)
          applyRule(start, end, rule, Collections.singletonList(deriv));
      } else {
        /* Apply non-cat unary rules by traversing through the subspans */
        int derivsCreated = 0;
        for (int i = start + 1; i < end; i++) {
          derivsCreated += applyFloatingRule(rule, start, end, chart[start][i], chart[i][end]);
          derivsCreated += applyFloatingRule(rule, start, end, chart[i][end], chart[start][i]);
        }

        /* If no derivs created, propagate up */
        if (derivsCreated == 0) {
          copyDerivs(chart[start][end - 1], chart[start][end]);
          if (start != numTokens - 1)
            copyDerivs(chart[start + 1][end], chart[start][end]);
        }
      }
    }
    // test prune
    Set<String> cellsPruned = new HashSet<>();
    for (Map.Entry<String, List<Derivation>> entry : chart[start][end].entrySet())
      pruneCell(cellsPruned, entry.getKey(), start, end, entry.getValue());
  }

  protected int applyFloatingRule(Rule rule, int start, int end, Map<String, List<Derivation>> first,
      Map<String, List<Derivation>> second) {
    List<Derivation> derivs1 = first.get(rule.rhs.get(0));
    List<Derivation> derivs2 = second.get(rule.rhs.get(1));

    if (derivs1 == null || derivs2 == null)
      return 0;

    int derivsCreated = 0;

    for (Derivation deriv1 : derivs1) {
      for (Derivation deriv2 : derivs2) {
        List<Derivation> children = new ArrayList<Derivation>();
        children.add(deriv1);
        children.add(deriv2);
        derivsCreated += applyRule(start, end, rule, children);
      }
    }

    return derivsCreated;
  }

  protected void copyDerivs(Map<String, List<Derivation>> source, Map<String, List<Derivation>> dest) {
    if (source == null || dest == null)
      return;

    for (String cat : source.keySet()) {
      List<Derivation> derivations = dest.get(cat);
      if (derivations == null)
        dest.put(cat, derivations = new ArrayList<>());

      /* add only if the formula not already present to ensure no duplicates */
      Set<Formula> formulas = new HashSet<Formula>();
      for (Derivation deriv : derivations)
        formulas.add(deriv.formula);

      for (Derivation deriv : source.get(cat)) {
        if (!formulas.contains(deriv.formula)) {
          derivations.add(deriv);
          formulas.add(deriv.formula);
        }
      }
    }
  }

  protected void addDerivs(List<Derivation> source, List<Derivation> dest) {
    if (dest == null || source == null)
      return;
    dest.addAll(source);
  }

  // -- Coarse state pruning --

  // Remove any (cat, start, end) which isn't reachable from the
  // (Rule.rootCat, 0, numTokens)
  public void keepTopDownReachable() {
    if (numTokens == 0)
      return;

    Set<String> reachable = new HashSet<>();
    collectReachable(reachable, Rule.rootCat, 0, numTokens);

    // Remove all derivations associated with (cat, start, end) that aren't
    // reachable.
    if (!Parser.opts.partialParsing){
	    for (int start = 0; start < numTokens; start++) {
	      for (int end = start + 1; end <= numTokens; end++) {
	        List<String> toRemoveCats = new LinkedList<>();
	        for (String cat : chart[start][end].keySet()) {
	          String key = catStartEndKey(cat, start, end);
	          if (!reachable.contains(key)) {
	            toRemoveCats.add(cat);
	          }
	        }
	        Collections.sort(toRemoveCats);
	        for (String cat : toRemoveCats) {
	          if (parser.verbose(4)) {
	            LogInfo.logs("Pruning chart %s(%s,%s)", cat, start, end);
	          }
	          chart[start][end].remove(cat);
	        }
	      }
	    }
    }
  }

  private void collectReachable(Set<String> reachable, String cat, int start, int end) {
    String key = catStartEndKey(cat, start, end);
    if (reachable.contains(key))
      return;

    if (!chart[start][end].containsKey(cat)) {
      // This should only happen for the root when there are no parses.
      return;
    }

    reachable.add(key);
    for (Derivation deriv : chart[start][end].get(cat)) {
      for (Derivation subderiv : deriv.children) {
        collectReachable(reachable, subderiv.cat, subderiv.start, subderiv.end);
      }
    }
  }

  private String catStartEndKey(String cat, int start, int end) {
    return cat + ":" + start + ":" + end;
  }

  // For pruning with the coarse state
  protected boolean coarseAllows(Trie node, int start, int end) {
    if (coarseState == null)
      return true;
    return SetUtils.intersects(node.cats, coarseState.chart[start][end].keySet());
  }

  protected boolean coarseAllows(String cat, int start, int end) {
    if (coarseState == null)
      return true;
    return coarseState.chart[start][end].containsKey(cat);
  }
  
  /**
   * Tries to extend the parsing of a non-parsable utterance using partial parsing 
   * and similarity with rules in the grammar to compute possible relevant derivations.
   * Adds the computed derivations to ex.predDerivations and this.predDerivations
   * @author Akshal Aniche
   */
  private void extendParsing() {
	  ArrayList<Derivation> matches = new ArrayList<Derivation>(ex.getTokens().size()); //keep track of which derivation corresponds to which category
	  
	  //abstract the utterance from the partially parsable fragments and track the corresponding derivations
	  List<String> rhs = getRHS(matches);
	  	  	   
	  //collect all the rules that match the utterance past the similarity threshold
	  final Map<Rule, Double> ruleSimilarityMap = new HashMap<Rule, Double>();
	  for (Rule rule : parser.allRules) {
		  double similarity = computeSimilarity(rhs, rule);
		  if (similarity > InteractiveBeamParser.opts.simMin) {
			  ruleSimilarityMap.put(rule, Double.valueOf(similarity));
		  }
	  }
	 
	  //Sort the rule in decreasing order of similarity
	  List<Rule> applicableRules = new ArrayList<Rule>(ruleSimilarityMap.keySet());
	  
	  if (applicableRules.size() > 1) { 
		  Collections.sort(applicableRules, 
				  	new Comparator<Rule>() {
				  		public int compare(Rule rule1, Rule rule2) {
				  			return (ruleSimilarityMap.get(rule2)).compareTo(ruleSimilarityMap.get(rule1)); 
				  		}
		  			}
		  ); 
	  }
	  
	  //Should we only consider the top 3 similar rules?
	  applicableRules = applicableRules.subList(0, Math.min(applicableRules.size(), 3));
	  
	  if (Parser.opts.verbose > 2) {
		  LogInfo.logs("Set of similar rules:");
		  LogInfo.logs(applicableRules.toString());
	  }
	  
	  List<Derivation> potentialDeriv = new ArrayList<Derivation>();
	  for (Rule rule : applicableRules) {
		  String matchingUtt = matchToRule(rule, matches);
		  if (Parser.opts.verbose > 2) 
			  LogInfo.logs("Utterance converted to %s", matchingUtt);
		  
		  potentialDeriv.addAll(getExtendedDerivationsFromUtterance(matchingUtt));
	  }
	  
	  if (Parser.opts.verbose > 2) {
		  LogInfo.logs("Potential derivations: ");
		  for (Derivation d : potentialDeriv) 
			  LogInfo.logs(d.toString());
	  }
	  
	  predDerivations.addAll(potentialDeriv);
	  if (ex.predDerivations == null)
		  ex.predDerivations = potentialDeriv;
	  else
		  ex.predDerivations.addAll(potentialDeriv);
	  return; 
  }
  
  /**
   * Takes an utterance obtained by matching a rule to generate derivations
   * might be an issue : returns multiple copies of the same derivation in the list
   * @author Akshal Aniche
   * @param utt utterance used to get derivations
   * @return List<Derivation> generated by parsing the utterance
   */
  private List<Derivation> getExtendedDerivationsFromUtterance(String utt) {
	  	//I couldn't access the session from InteractiveBeamParser to call InteractiveUtils.exampleFromUtterance
	//create example to pass to the parser  
	  Example.Builder b = new Example.Builder();
	  b.setId(ex.id);
	  b.setUtterance(utt);
	  b.setContext(ex.context);
	  Example exHead = b.createExample();
	  exHead.preprocess();
	  
	  //parse the utterance
	  parser.parse(params, exHead, false, true);
	  
	  if(Parser.opts.verbose > 2) {
		  LogInfo.logs("Utterance %s gave the following derivations %s", utt, exHead.predDerivations.toString());
	  }
	  
	  //null safe return 
	  if (exHead.predDerivations == null) 
		  return new ArrayList<Derivation>();
	  
	  for (Derivation d : exHead.predDerivations) {
		  d.parsingUtt = utt;
	  }
	  
	  //Should we only consider the top scoring derivations
//	  return exHead.predDerivations.subList(0,  Math.min(exHead.predDerivations.size(), 1));
	  	  
	  return exHead.predDerivations;
  }

  /**
   * Computes the similarity between the partially parsed utterance and a given rule's RHS 
   * similarity is a double between 0 and 1 inclusive.
   * Note: similarity = length of longest common subsequence / max(length of given rhs, length of rule's RHS)
   * @author Akshal Aniche
   * @param rhs partially parsed utterance abstracted using categories 
   * @param rule
   * @return similarity 
   * @throws RuntimeException if computed similarity is not within bounds
   */
  private double computeSimilarity(List<String> rhs, Rule rule) {
	  
	  if (Parser.opts.verbose > 5) {
		  LogInfo.logs("Computing similarity with rule %s", rule.toString());
	  }
	  
	  List<String> ruleRHS = new ArrayList<String>(rule.rhs);
	  
	  int uttLen = rhs.size();
	  int ruleLen = ruleRHS.size();
	  int longerLen = Math.max(uttLen, ruleLen);
	  if (longerLen <= 0) return 0.0; 	//invalid length
	  
	  //check if the categories (start with '$') in both RHS are equal
	  List<String> rhsCat = rhs.stream().filter(s -> s.startsWith("$")).collect(Collectors.toList());
	  List<String> ruleRhsCat = ruleRHS.stream().filter(s -> s.startsWith("$")).collect(Collectors.toList());
	  if (!rhsCat.equals(ruleRhsCat)) return 0.0;
	  
	  double similarity = 1.0;

	  //dynamic computation of longest common subsequence
	  int[][] subsequence = new int[uttLen + 1][ruleLen + 1];
	  
	  for (int i = 0; i <= uttLen; i++) 
		  subsequence[i][0] = 0;
	  for (int j = 0; j <= ruleLen; j++) 
		  subsequence[0][j] = 0;
	  for (int i = 1; i <= uttLen; i++) {
		  for (int j = 1; j <= ruleLen; j++) {
			  if (rhs.get(i-1).equals(ruleRHS.get(j-1))) 
				  subsequence[i][j] = 1 + subsequence [i-1][j-1];
			  else
				  subsequence[i][j] = Math.max(subsequence[i-1][j], subsequence[i][j-1]);
		  }
	  }
	  
	  int longestSubsequence = subsequence[uttLen][ruleLen];
	  
	  if (Parser.opts.verbose > 5) {
		  LogInfo.logs("Longest common subsequence length: %s", longestSubsequence);
	  }
	  
	  similarity = (double) longestSubsequence / (double) longerLen;
	  
	  if (similarity < 0.0 || similarity > 1.0)
		  throw new RuntimeException("Computational problem for similarity with rule " + rule.toString());
	  return similarity;
  }
  
  
  
  /**
   * Create a string that would match a given rule by replacing the categories in the rule's RHS
   * by the corresponding fragment of the original utterance
   * @author Akshal Aniche
   * @param rule
   * @param matches
   * @return matching String 
   * @throws IllegalArgumentException if the number of categories to replace is different from the number of tracked derivations
   */
  private String matchToRule (Rule rule, List<Derivation> matches) {
	 List<String> utterance = ex.getTokens();
	 List<String> rhs = rule.rhs;
	 List<String> categories = rhs.stream().filter(s -> s.startsWith("$")).collect(Collectors.toList());
  
	 if (matches.size() != categories.size()) 
		 throw new IllegalArgumentException ("There was an error converting categories into strings because of length mismatch.");
	 
	 List<String> converted = new ArrayList<String>();
	 
	 Iterator<Derivation> derivIter = matches.iterator();
	 Iterator<String> rhsIter = rhs.iterator();
	 while (rhsIter.hasNext()) {
		 String token = rhsIter.next();
		 //if it's a category, then replace it by the corresponding token
		 if (token.startsWith("$")) {
			 Derivation correspondingDeriv = derivIter.next();
			 for (int i = correspondingDeriv.start; i < correspondingDeriv.end; i++) {
				 //take the corresponding tokens in the utterance and add them to the converted list
				 converted.add(utterance.get(i));
			 }
		 }
		 //it's not a category, so there is no need to replace anything
		 else {
			 converted.add(token);
		 }
	 }
	 
	 //concatenate the converted List
	 String matchingUtterance = String.join(" ", converted);
	 return matchingUtterance;
  
  }
  
  
  /**
   * Transform a non parsable utterance into a list of Strings: categories corresponding to parsable fragments and non parsable tokens
   * @author Akshal Aniche
   * @param matches ArrayList of derivations in which to store the derivations corresponding to the parsable fragments
   * @return computed List<String>
   */
  private List<String> getRHS(ArrayList<Derivation> matches){
	  //base string
	  ArrayList<String> rhs = new ArrayList<String>(ex.getTokens());

	  for (int i = 0; i < rhs.size(); i++) 
		  matches.add(null);

	  if (Parser.opts.verbose > 4)
		  LogInfo.logs("before transforming: %s", rhs.toString());

	  
	  for (Derivation d: chartList) {
		  String cat = d.cat;
		  
		  //Ignore non-inducing categories and Action categories
		  if (!cat.toUpperCase().equals(cat) && !cat.startsWith("$Action")){
			  
			  //do not substitute $Areas by $Area
			  if (d.end - d.start == 1 && 
				  rhs.get(d.start) != null &&
				  rhs.get(d.start).equals("$Areas")) 
				  continue;
			  
		  	rhs.set(d.start, cat);
		  	matches.set(d.start, d);
		  	
		  	for (int i = d.start + 1; i < d.end; i++) {
		  		rhs.set(i, null);
		  		matches.set(i, null);
		  	}
		  }
	  }
	  
	  rhs.removeAll(Collections.singleton(null));
	  matches.removeAll(Collections.singleton(null));
	  
	  if (Parser.opts.verbose > 4) {
		  LogInfo.logs("After transforming: %s", rhs.toString());
		  LogInfo.logs("Corresponding derivations are: %s", matches.toString());
	  }
	  return rhs;
  }
}
