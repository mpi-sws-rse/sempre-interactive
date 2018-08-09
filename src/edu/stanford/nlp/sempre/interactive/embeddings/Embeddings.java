package edu.stanford.nlp.sempre.interactive.embeddings;

import java.util.List;

public class Embeddings {

	Dictionary dict;
	
	/**
	 * Returns the dot product of the two vectors
	 */
	static double dot(List<Double> l1, List<Double> l2) {
		if (l1.size() != l2.size())
			throw new IllegalArgumentException("The two vectors need to have the same dimension in a similarity computation.");
		double sum = 0.0;
		for (int i = 0; i < l1.size(); ++i)
			sum += l1.get(i) * l2.get(i);
		return sum;
	}
	
	/**
	 * Returns the cosine similarity between the two Words specified
	 */
	public static double sim(Word w1, Word w2) {
		if (w1.scalars == null || w2.scalars == null) 
			throw new IllegalArgumentException ("You cannot have one word vector be null in a similarity computation");
		return dot(w1.scalars, w2.scalars)/(w1.mag() * w2.mag());
	}
	
	
	public Embeddings(String embeddingsPath) {
		Reader reader = new Reader(embeddingsPath);
		this.dict = new Dictionary(reader);
	}
	
	/**
	 * Returns a Word (word vector representation) corresponding to the String word passed as argument
	 */
	public Word getWord (String word) {
		return dict.entries.getOrDefault(word, Word.nullWord);
	}
}