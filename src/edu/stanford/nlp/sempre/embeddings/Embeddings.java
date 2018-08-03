package edu.stanford.nlp.sempre.embeddings;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;
import java.util.Set;

import edu.stanford.nlp.sempre.Grammar;
import edu.stanford.nlp.sempre.Rule;

public class Embeddings {

	Dictionary dict;
	final Map<String,Set<Word>> categories = new HashMap<>();
	
	static double dot(List<Double> l1, List<Double> l2) {
		double sum = 0.0;
		for (int i = 0; i < l1.size(); ++i)
			sum += l1.get(i) * l2.get(i);
		return sum;
	}
	
	static double sim(Word w1, Word w2) {
		return dot(w1.scalars, w2.scalars)/(w1.mag() * w2.mag());
	}
	
	
	public Embeddings(String embeddingsPath, Collection<Rule> rules) {
	  
		Reader reader = new Reader(embeddingsPath);
		this.dict = new Dictionary(reader);
	  initializeCategories(rules, dict);
	}

	public String bestMatch(String category, String token) {
	  // What should happen if something is not found?
    Word tokenVec = dict.entries.getOrDefault(token, Word.nullWord);
    if (tokenVec.equals(Word.nullWord))
      return "ERR_NOT_FOUND";
		Set<Word> set = categories.get(category);
		double max = -1.1; // cosine min is -1.0
		Word maxWord = null;
		for (Word word : set) {
		  double sim = sim(word, tokenVec);
		  if (sim > max) {
		    maxWord = word;
		    max = sim;
		  }
		}
		return maxWord.name;
	}
	
//	public String bestMatchSafe(String category, String token) {
//	  new Random().nextInt()
//	}
	
	// TODO How should I properly handle when a word is not contained in the
	// embeddings? 
	void initializeCategories(Collection<Rule> rules, Dictionary dict) {
    for (Rule r : rules) {
      if (!r.isRhsTerminals() || r.rhs.get(0).equals(Grammar.UNK_TOKEN))
        continue;
      
      if (!categories.containsKey(r.lhs))
        categories.put(r.lhs, new HashSet<>());
      Set<Word> set = categories.get(r.lhs);
      Word w = dict.entries.get(r.rhs.get(0));
      if (w == null) {
        w = Word.zeroWordLike(r.rhs.get(0),
            dict.entries.entrySet().iterator().next().getValue());
      }
      set.add(w);
    }
	}


	
	/*
  static int testPair(Dictionary dict, String target, String token) {
    if (!dict.entries.containsKey(token))
      return 0;
    Word tokenVec = dict.entries.get(token);
		Set<Word> set = categories.get(target);
		double max = -1.1; // cosine min is -1.0
		Word maxWord = null;
		for (Word word : set) {
		  double sim = sim(word, tokenVec);
		  if (sim > max) {
		    maxWord = word;
		    max = sim;
		  }
		}
//		System.out.printf("%s %s %s\n", target, token, maxWord.name);
		if (target.equals(maxWord.name))
		  return 1;
		else
		  return -1;
  }
  
  static double processArray(List<Integer> list) {
    int correct = 0, total = 0;
    for (Integer x : list) {
      if (x == 0)
        continue;
      else if (x == 1)
        correct += 1;
      total += 1;
    }
    return 1.0 * correct / total;
  }
	
	static void runTest(Path dataPath, Path embeddingsPath) {
	  System.out.printf("Running %s\n", embeddingsPath);
		BufferedReader dataReader;
		BufferedReader embeddingsReader;
		BufferedWriter writer;
		
		Path outPath = Paths.get("..","results",
		    embeddingsPath.getFileName()+".results");

		Reader reader = new Reader(embeddingsPath.toString());
		Dictionary dict = new Dictionary(reader);
	  initializeCategories(dict);
		

		try {
			dataReader = new BufferedReader(new FileReader(
			    new File(dataPath.toString())));
//			embeddingsReader = new BufferedReader(new FileReader(
//			    new File(embeddingsPath.toString())));
			writer = new BufferedWriter(new FileWriter(
			    new File(outPath.toString())));

			Map<String, List<Integer>> results = new HashMap<>();
      while (dataReader.ready()) {
        String[] line = dataReader.readLine().split(",");
        int result = testPair(dict, line[0], line[1]);
        if (!results.containsKey(line[0])) {
          results.put(line[0], new ArrayList<>());
        } 
        results.get(line[0]).add(result);
      }
      for (Entry<String, List<Integer>> e : results.entrySet()) {
//        System.out.printf("%s: %.3f\n",
//            e.getKey(), processArray(e.getValue()));
        writer.write(e.getKey()+","+processArray(e.getValue())+"\n");
      }
      
      writer.close();
      dataReader.close();
//      embeddingsReader.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
	  
	}
	*/
	
	  /*
	static void getSynonyms(String readPath, String writePath) {
		gloss("build", RiWordNet.VERB);
		gloss("remove", RiWordNet.VERB);
		gloss("find", RiWordNet.VERB);
		gloss("box", RiWordNet.NOUN);
		gloss("sphere", RiWordNet.NOUN);
		gloss("large", RiWordNet.ADJ);
		gloss("medium", RiWordNet.ADJ);
		gloss("small", RiWordNet.ADJ);
	  
		BufferedReader reader;
		BufferedWriter writer;
		
		try {
			reader = new BufferedReader(new FileReader(new File(readPath)));
			writer = new BufferedWriter(new FileWriter(new File(writePath)));

      while (reader.ready()) {
        String[] line = reader.readLine().split(",");
        String[] synonyms = wordnet.getSynonyms(Integer.parseInt(line[1]));
        for (int i = 0; i < synonyms.length; ++i) {
          writer.write(line[0]+","+synonyms[i]+"\n");
        }
      }
      writer.close();
      reader.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
		
	}
	
	static void gloss(String word, String pos) {
		  System.out.printf("Glossing: %s\n", word);
//		System.out.println(
//		    Arrays.asList(wordnet.getSenseIds("add", RiWordNet.VERB)));
		int[] ids = wordnet.getSenseIds(word, pos);
		for (int i = 0; i < ids.length; ++i) {
		  System.out.println(ids[i]);
		  System.out.println(Arrays.asList(wordnet.getSynonyms(ids[i])));
		  System.out.println();
		}

	  
	}
	
	static void nearestNeighbors(Dictionary dict, Word t1, Word t2) {
		List<Map.Entry<Word,Double>> distList = new ArrayList<>();
		for (Map.Entry<String, Word> e : dict.entries.entrySet()) {
			double d1 = distance(t1.scalars, e.getValue().scalars);
			double d2 = distance(t2.scalars, e.getValue().scalars);
			double dist = 5*Math.pow(d1-d2, 2) + d1 + d2;
			distList.add(new AbstractMap.SimpleEntry<Word, Double>
				(e.getValue(), dist));
		}
		
		Collections.sort(distList, Comparator.comparingDouble(
				(x) -> x.getValue()));

		for (int i = 0; i < 10; ++i) {
			System.out.println(distList.get(i).getKey());
		}
		
		System.out.println("\nDone.");
		
	}
	
	static double sumSquares(List<Double> l) {
		return l.stream().reduce(0.0, (a, b) -> a + b*b);
	}
	
	static double distance(List<Double> l1, List<Double> l2) {
		double dist = 0.0;
		for (int i = 0; i < l1.size(); ++i)
			dist += Math.pow(l1.get(i) - l2.get(i), 2);
		return dist;
	}

	static void scaleWith(Word w, List<Double> vals) {
		for (int i = 0; i < w.scalars.size(); ++i)
			w.scalars.set(i, w.scalars.get(i) * vals.get(i));
	}
	
	static List<Double> squareDiffs(List<Double> l1, List<Double> l2) {
		ArrayList<Double> res = new ArrayList<>();
		for (int i = 0; i < l1.size(); i++) 
			res.add(Math.pow(l1.get(i) - l2.get(i), 2));
		return res;
	}
	
	static Word combine(Word w1, Word w2) {
		ArrayList<Double> vals = new ArrayList<>();
		for (int i = 0; i < w1.scalars.size(); ++i)
			vals.add((w1.scalars.get(i) + w2.scalars.get(i))/2);
		Word w = new Word(w1.name + " " + w2.name, vals);
		return w;
	}
	   */
}
