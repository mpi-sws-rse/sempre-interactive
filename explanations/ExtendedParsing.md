Author: Akshal Aniche

# Extend the parsing of a unparsable utterance
This implementation allows for extending the parsing of a unparsable utterance to similar rules without needing to match it exactly.
It needs the option 'Parser.partialParsing' to be set to 'true'.
Relevant functions are 'extendParsing', 'getRHS', 'computeSimilarity', 'matchToRule' and 'getExtendedDerivations' in 'InteractiveBeamParser.java', 

## High-level strategy
The skeleton of the extended parsing is in 'extendParsing'.

### Abstract from the utterance using partial derivations
If the utterance can be parsed partially, we want to abstract from the parsable tokens and generalize the utterance.
The function 'getRHS()' replaces the tokens that are part of the partial derivations by the category of the derivation. (Non-inducing categories and $Action are not taken into account)
It also keeps track of the derivations that match the categories, to be able to recover the tokens later on.

### Finding similar rules
Iterating through all of the rules in 'extendParsing', the RHS obtained from 'getRHS' is compared to the RHS of each rule using the function 'computeSimilarity'.
It returns a double representing the level of similarity of the rule to the abstracted RHS. 
A rule needs to use the same categories as the categories in RHS, and needs to have similarity higher than 'InteractiveBeamParser.opts.simMin' to be considered similar to RHS.

### Creating a parsable utterance to match the rule
After finding the set of all similar rules, we use the function 'matchToRule' to substitute the categories in the rule's RHS by the tokens that corresponded to the partial derivations in the unparsable utterance. This uses the list of derivations we tracked in 'getRHS'.
We will thus create a string that will be parsable by the rule used to create it.

### Finding extended derivations
We parse the created utterance (that we know to be parsable) and obtain a set of derivations that are sent to the user.

## Inducing a definition
We track the utterance that was used to create an extended derivation as well as the original unparsable utterance.
If the user accepts an interpretation obtained from an extended parsing, then a request is sent to the back end to define the unparsable utterance using the parsable similar utterance as body.

## Tracking the utterances
The field parsingUtt in 'Derivation' serves to track the utterance that was parsed to get the derivation, but is only set for derivations obtained through extended parsing.
That utterance is added as a field to the candidate sent in the response to the user query.

The unparsable utterance is stored in the React state, and it serves as a flag to signal that a definition should be induced when accepting an interpretation.


