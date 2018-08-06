package edu.stanford.nlp.sempre.interactive;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Set;
import java.util.Comparator;
import java.util.Collections;
import java.util.HashSet;
import java.util.stream.Collectors;

import org.testng.collections.Lists;

import com.google.common.collect.ImmutableList;

import edu.stanford.nlp.sempre.ActionFormula;
import edu.stanford.nlp.sempre.BeamParser;
import edu.stanford.nlp.sempre.Derivation;
import edu.stanford.nlp.sempre.Example;
import edu.stanford.nlp.sempre.Formula;
import edu.stanford.nlp.sempre.Formulas;
import edu.stanford.nlp.sempre.IdentityFn;
import edu.stanford.nlp.sempre.Json;
import edu.stanford.nlp.sempre.Master;
import edu.stanford.nlp.sempre.Params;
import edu.stanford.nlp.sempre.Parser;
import edu.stanford.nlp.sempre.Rule;
import edu.stanford.nlp.sempre.SemanticFn;
import edu.stanford.nlp.sempre.Session;
import fig.basic.LispTree;
import fig.basic.LogInfo;
import fig.basic.Option;
import fig.basic.Ref;

/**
 * Utilities for interactive learning
 *
 * @author sidaw
 */


public final class InteractiveUtils {
  public static class Options {
    @Option(gloss = "use the best formula when no match or not provided")
    public boolean useBestFormula = false;

    @Option(gloss = "path to the citations")
    public String citationPath;

    @Option(gloss = "verbose")
    public int verbose = 0;
  }

  public static Options opts = new Options();

  public InteractiveUtils() {
  }

  // dont spam my log when reading things in the beginning...
  public static boolean fakeLog = false;

  public static Derivation stripDerivation(Derivation deriv) {
    while (deriv.rule.sem instanceof IdentityFn) {
      deriv = deriv.child(0);
    }
    return deriv;
  }
  
  public static Example exampleFromUtterance(String utt, Session session) {
    Example.Builder b = new Example.Builder();
    b.setId(session.id);
    b.setUtterance(utt);
    b.setContext(session.context);
    Example ex = b.createExample();
    ex.preprocess();
    return ex;
  }

  public static Derivation stripBlock(Derivation deriv) {
    if (opts.verbose > 0)
      LogInfo.logs("StripBlock %s %s %s", deriv, deriv.rule, deriv.cat);
    while ((deriv.rule.sem instanceof BlockFn || deriv.rule.sem instanceof IdentityFn) && deriv.children.size() == 1) {
      deriv = deriv.child(0);
    }
    return deriv;
  }

  static class Packing {
	    List<Derivation> packing;
	    double score;
	    
	    public Packing(double score, List<Derivation> packing) {
	      this.score = score;
	      this.packing = packing;
	    }

	    @Override
	    public String toString() {
	      return this.score + ": " + this.packing.toString();
	    }
	  }


  private static int blockingIndex(List<Derivation> matches, int end) {
	    return matches.stream().filter(d -> d.end <= end).map(d -> d.start).max((s1, s2) -> s1.compareTo(s2))
	        .orElse(Integer.MAX_VALUE / 2);
	  }
  
  

  /**
   * function that calculates maximal packings for the interval (start, end) in a recursive manner
   * <p>
   * the set of maximal packings M(start,end) is equal to {interval_list[start][end]} union union_i M(start,i) + M(i,end)
   * here, the plus operator is:  [ [x, y], [z] ] + [[a], [b]] = [ [x, y, a], [x, y, b], [z, a], [z, b] ]
   * 
   * 
   */
  private static List<List<Derivation>> getTheFValue(List<List<Derivation>>[][] f_table, Derivation[][] intervalTable, int start, int end){
	  
	  // if f_table was already filled, return the result
	  if (f_table[start][end] != null){
		
		  return f_table[start][end];
	  }
	  else{
		  List<List<Derivation>> f_value = new ArrayList<List<Derivation>>();
		  List<Derivation> initialCandidate = new ArrayList<Derivation>();
		  Derivation wholeDerivation = intervalTable[start][end];
		  // if there is a parse of the whole (start, end) interval, add it to maximal derivations
		  if (wholeDerivation != null){
			  initialCandidate.add(wholeDerivation);
			  f_value.add(initialCandidate);
		  }
		  
		  // iterate over all i between start and end
		  for (int i = start + 1; i < end; ++i){
			  
			  
			  List<List<Derivation>> leftHandSideCandidates = getTheFValue(f_table, intervalTable, start, i);
			  
			  List<List<Derivation>> rightHandSideCandidates = getTheFValue(f_table, intervalTable, i, end);
			  
			  // treat specially depending on the emptiness of left or right candidates
			  if (rightHandSideCandidates.size() == 0 && leftHandSideCandidates.size() == 0){
				  continue;
			  }
			  else if (rightHandSideCandidates.size() != 0 && leftHandSideCandidates.size() == 0){
				  for (List<Derivation> rld :rightHandSideCandidates){
					  if (!f_value.contains(rld)){
						  f_value.add(rld);
					  }
				  }
			  }
			  
			  else if (leftHandSideCandidates.size() != 0 && rightHandSideCandidates.size() == 0){
				  for (List<Derivation> lld : leftHandSideCandidates){
					  if (!f_value.contains(lld)){
						  f_value.add(lld);
					  }
				  }
			  }
			  
			  else{
			  
				  for (List<Derivation> lld : leftHandSideCandidates){
					  for (List<Derivation> rld : rightHandSideCandidates){
						  List<Derivation> combinationCandidate = new ArrayList<Derivation>();
						  combinationCandidate.addAll(lld);
						  combinationCandidate.addAll(rld);
						  if (!f_value.contains(combinationCandidate)){
							  f_value.add(combinationCandidate);
						  }
					  }
				  }
			  }
			  
		  }
		  
		  f_table[start][end] = f_value;
		  if (opts.verbose > 2){
			  LogInfo.logs("getTheFValue: returning for (%d, %d): %s", start, end, f_value.toString());
		  }
		  return f_value;
	  }
  }

  
  
  /**
   * the function that returns all maximal packings - sets of non overlapping partial derivations where adding any other partial derivation would make
   * the set overlapping.
   * @param partialParses list of all partial parses of the utterance (potentially overlapping)
   * @param length length of the utterance
   * @return maximalPackings
   */
  public static List<List<Derivation>> allMaximalPackings(List<Derivation> partialParses, int length){
	  
	  
	  // remove all CAPS categories -> e.g. KEYWORD_TOKEN
	  partialParses = partialParses.stream().filter(s -> ! s.getCat().equals(s.getCat().toUpperCase())).collect(Collectors.toList());
	  if (opts.verbose > 1){
		  LogInfo.logs("partialParses after filtering");
		  for (Derivation d : partialParses){
			  LogInfo.logs("derivation %s", d.toSimpleString());
			  d.printDerivationRecursively();
		  }
	  }
	  
	  // used for storing derivations based on their (start, end) range
	  // intervals are stored as (closed, open), that's why we need length+1 as a last one
	  Derivation[][] interval_list = new Derivation[length+1][length+1];
	  
	  // used for memorizing all maximal packings in the range (start, end)
	  List<List<Derivation>>[][] f_table = new ArrayList[length+1][length+1];
	  for (int i = 0; i < length+1; i++){		  
		  f_table[i] = new ArrayList[length+1];
	  }
	  
	  // if derivations are over the exactly same range, we'll keep in the interval_list only one of them (the one with the largest tree)
	  // Therefore, we first sort it by the tree size (descending) and then include only the first derivation that occurs per range
	  Collections.sort(partialParses, 
			  new Comparator<Derivation>(){
		  @Override
		  public int compare(Derivation d1, Derivation d2){
			  if (d1.derivCategoriesDF().size() > d2.derivCategoriesDF().size()){
				  return -1;
			  }			  
			  else {
				  return 1;
			  }
		  }
	  }
	  );

	  
	  
	  
	  for (Derivation d : partialParses){
		  Derivation initial_I_J = interval_list[d.start][d.end];
		  if (initial_I_J == null){
			  interval_list[d.start][d.end] = d;
		  }
	  }
	  
	  if (opts.verbose > 1){
		  for (int i = 0; i < length + 1; ++i){
			  for (int j = 0; j < length + 1; ++j){
				  if (interval_list[i][j] != null){
					  LogInfo.logs("partialParses at (%d, %d):", i, j);
					  Derivation d =interval_list[i][j]; 
					  LogInfo.logs("derivation %s", d.toSimpleString());
				  }
			  }
		  }
	  }
	  // call the recursive function getTheFValue with arguments 0 - length (e.g. the set of maximal packings of range starting at 0, ending at length) 
	  List<List<Derivation>> maximalPackings = getTheFValue(f_table, interval_list, 0, length); 
	  if (opts.verbose > 0){
		  LogInfo.logs("maximalPackings = %s", maximalPackings);
	  }
	  return maximalPackings;
  }
  
  public static List<Derivation> bestPackingDP(List<Derivation> matches, int length) {
		
		
	    List<Packing> bestEndsAtI = new ArrayList<>(length + 1);
	    List<Packing> maximalAtI = new ArrayList<>(length + 1);
	    bestEndsAtI.add(new Packing(Double.NEGATIVE_INFINITY, new ArrayList<Derivation>()));
	    maximalAtI.add(new Packing(0.0, new ArrayList<Derivation>()));

	    @SuppressWarnings("unchecked")
	    List<Derivation>[] endsAtI = new ArrayList[length + 1];

	    for (Derivation d : matches) {
	    	
	      List<Derivation> derivs = endsAtI[d.end];
	      derivs = derivs != null ? derivs : new ArrayList<>();
	      derivs.add(d);
	      endsAtI[d.end] = derivs;
	    }

	    for (int i = 1; i <= length; i++) {
	      // the new maximal either uses a derivation that ends at i, plus a
	      // previous maximal
	      Packing bestOverall = new Packing(Double.NEGATIVE_INFINITY, new ArrayList<>());
	      Derivation bestDerivI = null;
	      if (endsAtI[i] != null) {
	        for (Derivation d : endsAtI[i]) {
	          double score = d.getScore() + maximalAtI.get(d.start).score;
	          if (score >= bestOverall.score) {
	            bestOverall.score = score;
	            bestDerivI = d;
	          }
	        }
	        List<Derivation> bestpacking = new ArrayList<>(maximalAtI.get(bestDerivI.start).packing);
	        bestpacking.add(bestDerivI);
	        bestOverall.packing = bestpacking;
	      }
	      bestEndsAtI.add(i, bestOverall);

	      // or it's a previous bestEndsAtI[j] for i-minLength+1 <= j < i
	      for (int j = blockingIndex(matches, i) + 1; j < i; j++) {
	        if (bestEndsAtI.get(j).score >= bestOverall.score)
	          bestOverall = bestEndsAtI.get(j);
	      }
	        //LogInfo.logs("maximalAtI[%d] = %f: %s, BlockingIndex: %d", i, bestOverall.score, bestOverall.packing,blockingIndex(matches, i));
	            
	      if (bestOverall.score > Double.NEGATIVE_INFINITY)
	        maximalAtI.add(i, bestOverall);
	      else {
	        maximalAtI.add(i, new Packing(0, new ArrayList<>()));
	      }
	    }
	    return maximalAtI.get(length).packing;
	  }
  
  public static Derivation derivFromUtteranceAndFormula(String utterance, Formula formula, Parser parser, Params params, Session session){

	  Derivation foundDerivation = null;
	  List<Derivation> allDerivs = new ArrayList<>();
	  Formula targetFormula = formula;
	  
	  String utt = utterance;
	  	  
	  Example exHead = InteractiveUtils.exampleFromUtterance(utt, session);
	  
	  if (exHead.getTokens() == null || exHead.getTokens().size() == 0)
	      throw BadInteractionException.headIsEmpty(utt);
	    
	  InteractiveBeamParserState state = ((InteractiveBeamParser)parser).parseWithoutExecuting(params, exHead, false);
	  
	  
	  
	
	  boolean found = false;
	  for (Derivation d : exHead.predDerivations) {
		if (opts.verbose > 2){
		  LogInfo.logs("Deriv from utteranceAnd formula, predDerivations. considering: %s", d.formula.toString());
		}
		if (d.formula.equals(targetFormula)) {
		  found = true;
		  foundDerivation = stripDerivation(d);
		  break;
		}
	  }
	  if (!found && !formula.equals("?")) {
		  LogInfo.errors("matching formula not found: %s :: %s", utt, formula);
	  }
	  // just making testing easier, use top derivation when we formula is not
	  // given
	  if (!found && exHead.predDerivations.size() > 0 && (formula.equals("?") || formula == null || opts.useBestFormula))
		  foundDerivation = stripDerivation(exHead.predDerivations.get(0));
	  else if (!found) {
		  Derivation res = new Derivation.Builder().formula(targetFormula)
	        // setting start to -1 is important,
	// which grammarInducer interprets to mean we do not want partial
	// rules
				  .withCallable(new SemanticFn.CallInfo("$Action", -1, -1, null, new ArrayList<>())).createDerivation();
	    foundDerivation = res;
	  }
		    
	  if (foundDerivation != null && opts.verbose > 1){
		  LogInfo.logs("derivFromUtteranceAndFormula: returning deriv  %s", foundDerivation);
	  }
	 return foundDerivation;
	
		  
}
	  
  public static List<Derivation> derivsfromJson(String jsonDef, Parser parser, Params params,
      Ref<Master.Response> refResponse, Session session) {
    @SuppressWarnings("unchecked")
    List<Object> body = Json.readValueHard(jsonDef, List.class);
    // string together the body definition
    List<Derivation> allDerivs = new ArrayList<>();
    int numFailed = 0;
    for (Object obj : body) {
      @SuppressWarnings("unchecked")
      List<String> pair = (List<String>) obj;
      String utt = pair.get(0);
      String formula = pair.get(1);
      

      if (formula.equals("()")) {
        LogInfo.logs("Error: Got empty formula");
        continue;
      }
      Formula targetFormula = Formulas.fromLispTree(LispTree.proto.parseFromString(formula));
      Derivation newDerivation = derivFromUtteranceAndFormula(utt, targetFormula, parser, params, session);
      if (newDerivation != null){
    	  allDerivs.add(newDerivation);
      }
     }
     
   
   return allDerivs;
  }

  public static List<String> utterancefromJson(String jsonDef, boolean tokenize) {
    @SuppressWarnings("unchecked")
    List<Object> body = Json.readValueHard(jsonDef, List.class);
    // string together the body definition
    List<String> utts = new ArrayList<>();
    for (int i = 0; i < body.size(); i++) {
      Object obj = body.get(i);
      @SuppressWarnings("unchecked")
      List<String> pair = (List<String>) obj;
      String utt = pair.get(0);

      Example.Builder b = new Example.Builder();
      // b.setId("session:" + sessionId);
      b.setUtterance(utt);
      Example ex = b.createExample();
      ex.preprocess();

      if (tokenize) {
        utts.addAll(ex.getTokens());
        if (i != body.size() - 1 && !utts.get(utts.size() - 1).equals(";"))
          utts.add(";");
      } else {
        utts.add(String.join(" ", ex.getTokens()));
      }

    }
    return utts;
  }

  public static synchronized void addRuleInteractive(Rule rule, Parser parser) {
    if (parser instanceof InteractiveBeamParser) {
      parser.addRule(rule);
    } else {
      throw new RuntimeException("interactively adding rule not supported for parser " + parser.getClass().toString());
    }
  }
  
  /**
   * Deletes a rule interactively from the parser
   * @param rule to be deleted
   * @param parser from where to delete it
   */
  public static synchronized void removeRuleInteractive(Rule rule, Parser parser) {
	    if (parser instanceof InteractiveBeamParser) {
	      parser.removeRule(rule);
	    } else {
	      throw new RuntimeException("interactively removing rule not supported for parser " + parser.getClass().toString());
	    }
	  }

  static Rule blockRule(ActionFormula.Mode mode) {
    BlockFn b = new BlockFn(mode);
    b.init(LispTree.proto.parseFromString("(BlockFn sequential)"));
    return new Rule("$Action", Lists.newArrayList("$Action", "$Action"), b);
  }

  public static Derivation combine(List<Derivation> children) {
    ActionFormula.Mode mode = ActionFormula.Mode.sequential;
    if (children.size() == 1) {
      return children.get(0);
    }
    Formula f = new ActionFormula(mode, children.stream().map(d -> d.formula).collect(Collectors.toList()));
    Derivation res = new Derivation.Builder().formula(f)
        // setting start to -1 is important,
        // which grammarInducer interprets to mean we do not want partial rules
        .withCallable(new SemanticFn.CallInfo("$Action", -1, -1, blockRule(mode), ImmutableList.copyOf(children)))
        .createDerivation();
    return res;
  }

  public static String getParseStatus(Example ex) {
    return GrammarInducer.getParseStatus(ex).toString();
  }
  
  public static enum AuthorDescription{
	  Self, Other, None;
  }
  
  public static boolean otherAuthors(Derivation d, String id) {
	  
	  if (d.rule.source != null && d.rule.source.uid != id) {
		  
		  return true;
	  }
	  else {
		  for (Derivation child : d.children) {
			  if (otherAuthors(child, id) == true) {
				  return true;
			  }
		  }
		  return false;
	  }
	  
  }
  
  // if there is at least one other author involved in the derivations, it is counted as "using other author's definitions"
  // if it is a core-language command, then the author is None. If it is induced, but only one author (the one who sent a command), then it is Self
  public static AuthorDescription getAuthorDescription(Example ex){
	  
	  for (Derivation d : ex.getPredDerivations()) {
		 if (d.isInduced() == false) {
			 return AuthorDescription.None;
		 }
		 else if (otherAuthors(d, ex.id)) {
			  return AuthorDescription.Other;
		 }
		 else {
			  return AuthorDescription.Self;
		  }
	  }
	  return AuthorDescription.None;
  }

  public static void cite(Derivation match, Example ex) {
    CitationTracker tracker = new CitationTracker(ex.id, ex);
    tracker.citeAll(match);
  }
}
