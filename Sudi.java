import java.io.*;
import java.util.*;

/**
 * Parsing natural language and assigning part of speech tags via Viterbi algorithm and hidden markov models
 * @author Sam Lefkowitz
 * May 24, 2021.  Updated file paths February 2, 2023
 */

public class Sudi {
    private static final double unseenObsScore = -100;       // score for unseen observations
    private static final String version = "partitionedEC";      // specify the version method to use: "consoleTest" | "fileTest"  | "partitionedEC"
    private static Scanner in;

    /**
     * Helper method takes a string and will return an array of the words in the sentence in a consistent format
     *
     * @param line The original string
     * @return Array of words that make up the string converted to lowercase and with whitespace removed
     */
    private static String[] processString(String line) {
        // process the lines
        String[] words = line.split("\\ ");            // split line into words based on a single space
        for (int i = 0; i < words.length; i++) {
            // process each word
            words[i] = words[i].trim();
            words[i] = words[i].toLowerCase();
        }
        return words;
    }

    /**
     * Takes 2 keys. The current state and the value that maps into the score. Keeps a running total of number of values
     * in a given row.  This value is accessible with 'TOTAL'.  All words and tags are lower case so no collision issues
     * TOTAL is an inner map key. State gives the row
     *
     * @param map   the map that is being generated
     * @param state current state.  Key to internal map
     * @param iKey  the internal map key.  string value we will update
     */
    private static void updateMaps(Map<String, Map<String, Integer>> map, String state, String iKey) {
        if (map.containsKey(state)) {
            map.get(state).put("TOTAL", map.get(state).get("TOTAL") + 1);     // total update
            if (map.get(state).containsKey(iKey)) {      // check if there exist a value for the internal key
                map.get(state).put(iKey, map.get(state).get(iKey) + 1);
            } else {      // new entry with value of 1
                map.get(state).put(iKey, 1);
            }
        } else {      // create new entry in outer map.  Update new inner map internal key entry and associated value of 1
            map.put(state, new HashMap<>());
            map.get(state).put(iKey, 1);        // put value of 1 for internal map key
            map.get(state).put("TOTAL", 1);
        }
    }

    /**
     * Helper method which takes maps of strings to strings to integers-- the occurrence counts for tag to word or next
     * pos and generates the probability map in terms of the natural log
     *
     * @param map Map of counts used to generate log probability map
     * @return The log prob maps to be used in viterbi algorithm
     */
    private static Map<String, Map<String, Double>> createLogProbMap(Map<String, Map<String, Integer>> map) {
        Map<String, Map<String, Double>> rMap = new HashMap<>();
        // loop through once to get row totals
        for (String state : map.keySet()) {
            rMap.put(state, new HashMap<>());
            for (String iKey : map.get(state).keySet()) {
                if (!iKey.equals("TOTAL")) {
                    // create log prob entry.  avoid entries for total count
                    rMap.get(state).put(iKey, Math.log((double) (map.get(state).get(iKey)) / (double) (map.get(state).get("TOTAL"))));
                }
            }
        }
        return rMap;
    }

    /**
     * Method used to create the both the transition and obs probability models
     *
     * @param trainingInputs an array of buffered readers that contain the tags in position 0 and the word obs in position 1
     * @return probability maps
     * @throws Exception IO exception
     */
    public static Map<String, Map<String, Double>>[] trainModel(BufferedReader[] trainingInputs) throws Exception {
        if (trainingInputs.length != 2) {
            System.out.println("Please input training documents as an array with the tag file in position 0 and the corresponding sentences in position 1");
            return null;
        }
        // initialize maps to return
        Map<String, Map<String, Integer>> transMap = new HashMap<>();
        Map<String, Map<String, Integer>> obsMap = new HashMap<>();
        // strings to hold lines that are read
        String transLine;
        String obsLine;
        // read line by line
        while ((transLine = trainingInputs[0].readLine()) != null && (obsLine = trainingInputs[1].readLine()) != null) {
            String prev = "#";  // initialize string to hold previous tag for transition. previous of first is start #
            String[] transTags = processString(transLine);
            String[] obsWords = processString(obsLine);
            for (int i = 0; i < transTags.length; i++) {     // cannot do for each loop: Need to index into same place on each array
                // update maps
                updateMaps(obsMap, transTags[i], obsWords[i]);     // trans word is the tag--State.  obsWord is observed word
                updateMaps(transMap, prev, transTags[i]);      // prev is the previous state.  current state is the next of previous. update transitions

                // store last word for next transition occurrence
                prev = transTags[i];
            }
        }
        // generate log prob maps to be returned add these maps to the returned array.
        // Return in order documents are put in--Tag transition, Word observation
        return new Map[]{createLogProbMap(transMap), createLogProbMap(obsMap)};
    }

    /**
     * This method implements the viterbi algorithm given a observation score log prob map and a similar transition map
     * It does the scoring line by line on a single sentence
     *
     * @param line       The sentence of interest
     * @param obsScore   The log prob map for the observations
     * @param transScore The log prob map for the transition
     * @return The most likely set of tags
     * @throws Exception IO exception in reading the documents
     */
    public static String[] viterbiPOS(String line, Map<String, Map<String, Double>> obsScore, Map<String, Map<String, Double>> transScore) throws Exception {
        // process the line
        String[] words = processString(line);
        // initialize data structures for viterbi
        Set<String> currStates = new HashSet<>();      // Maps potential currState -> map of potential next state & corresponding score
        Map<String, Double> currScores = new HashMap<>();                   // Maps potential currState -> score
        ArrayList<Map<String, String>> backpointer = new ArrayList<>();     // Array list of maps to keep track of backpointing path
        currStates.add("#");
        currScores.put("#", 0.0);
        for (int i = 0; i < words.length; i++) {
            Set<String> nextStates = new HashSet<>();
            Map<String, Double> nextScores = new HashMap<>();
            Map<String, String> bpMap = new HashMap<>();
            for (String currState : currStates) {
                // for each possible current state
                if (transScore.get(currState) == null) {
                    continue;
                } // no transition observed check the next potential current state
                for (String potNext : transScore.get(currState).keySet()) {
                    // consider potential next states by computing next score for each potential next
                    double nextScore = currScores.get(currState);          // score to current state
                    nextScore += transScore.get(currState).get(potNext);  // score to next state
                    if (obsScore.get(potNext).containsKey(words[i])) {    // observation scores exists for potential POS
                        nextScore += obsScore.get(potNext).get(words[i]);
                    } else {
                        nextScore += unseenObsScore;
                    }
                    if (!nextStates.contains(potNext) || nextScores.get(potNext) < nextScore) {
                        // when condition met this path is the most likely for the potential next state.
                        // points to current state for backpointer as best path.  update the next scores to keep track of path
                        nextScores.put(potNext, nextScore);
                        nextStates.add(potNext);
                        bpMap.put(potNext, currState);
                    }
                }
            }
            // move forward to next word
            currStates = nextStates;
            currScores = nextScores;
            backpointer.add(bpMap);
        }

        // best path for each potential POS has been checked.  Determine best overall score in final state
        Double bestScore = null;
        String bestState = null;
        for (String finalState : currStates) {
            if (bestScore == null || currScores.get(finalState) > bestScore) {
                bestScore = currScores.get(finalState);
                bestState = finalState;
            }
        }
        // back track and assign tags
        int i = words.length;     // start in last position
        String[] prevState = new String[words.length];
        while (!bestState.equals("#")) { // until the start is found.  put into array so # is at position -1 and not included
            prevState[i - 1] = bestState;
            i--;
            bestState = backpointer.get(i).get(bestState);
        }
        return prevState;
    }

    /**
     * Method that will instantiate a scanner object to take command line inputs in the console and run the viterbi method
     * to assign tags.
     *
     * @param obsProbMap   The model stored as maps for state to obs
     * @param transProbMap The model as stored in maps state to next state map
     * @throws Exception
     */
    private static void tagFromConsole(Map<String, Map<String, Double>> obsProbMap, Map<String, Map<String, Double>> transProbMap) throws Exception {
        in = new Scanner(System.in);
        boolean play = true;
        while (play) {
            System.out.println("Write sentence to be tagged. Write 'QUIT' to exit");
            String line = in.nextLine();
            String out = "";
            if (line.equals("QUIT")) play = false;
            else {
                String[] tags = viterbiPOS(line, obsProbMap, transProbMap);
                String[] words = processString(line);
                for (int i = 0; i < tags.length; i++) {
                    out += words[i] + "/" + tags[i] + " ";
                }
                System.out.println(out);
            }
        }
    }

    /**
     * Tests the tagging of test sentences based on the model in the maps passed as parameter
     *
     * @param test         test file readers in array of buffered readers.  Sentences to be tagged in position 1. Tags in position 0
     * @param obsProbMap   the observation log prob map
     * @param transProbMap the transition log prob map
     * @throws Exception IO exceptions
     */
    private static void fileTestTags(BufferedReader[] test, Map<String, Map<String, Double>> obsProbMap, Map<String, Map<String, Double>> transProbMap) throws Exception {
        String testLine, testCheck;
        int correctTags = 0;
        int checkedTags = 0;
        while ((testLine = test[1].readLine()) != null && (testCheck = test[0].readLine()) != null) {
            String[] viterbiResult = viterbiPOS(testLine, obsProbMap, transProbMap);
            String[] answers = processString(testCheck);
            if (viterbiResult.length != answers.length) {
                System.out.println("Resulting tags are not equal in length");
            } else {
                for (int i = 0; i < viterbiResult.length; i++) {
                    if (viterbiResult[i].equals(answers[i])) correctTags += 1;
                    //else System.out.println(processString(testLine)[i] + ": " +viterbiResult[i] + " " + answers[i]);  // identify incorrect tags. many of these appear to be unseen for the simple training
                }
                checkedTags += viterbiResult.length;
            }
        }
        System.out.println(correctTags + "/" + checkedTags + " are correct.");
    }

    /**
     * Main method to exercise the code written above and determine the number of tags correctly identified
     *
     * @param args
     */
    public static void main(String[] args) {
        try {
            // open readers in array
            BufferedReader[] train = new BufferedReader[]{new BufferedReader(new FileReader("trainingTexts/brown-train-tags.txt")), new BufferedReader(new FileReader("trainingTexts/brown-train-sentences.txt"))};

            //generate the maps from training
            Map<String, Map<String, Double>>[] temp = trainModel(train);       // store return value
            Map<String, Map<String, Double>> transProbMap = temp[0];
            Map<String, Map<String, Double>> obsProbMap = temp[1];

            // close the readers
            for (BufferedReader bufferedReader : train) {
                bufferedReader.close();
            }
            if (version.equals("consoleTest")) {
                tagFromConsole(obsProbMap, transProbMap);
            } else if (version.equals("fileTest")) {
// open test readers in array
                BufferedReader[] test = new BufferedReader[]{new BufferedReader(new FileReader("trainingTexts/brown-test-tags.txt")), new BufferedReader(new FileReader("trainingTexts/brown-test-sentences.txt"))};


                fileTestTags(test, obsProbMap, transProbMap);
                // close the readers
                for (BufferedReader bufferedReader : test) {
                    bufferedReader.close();
                }
            } else if (version.equals("partitionedEC")) {
                partitionedTest(5, false);
            } else {
                System.out.println("Please set 'version' static instance variable to desired test method: 'fileTest', 'hardcodedTest', 'consoleTest', 'generativeEC', 'partitionedEC'");
            }
        } catch (Exception e) {
            System.out.println(e);
        }
    }

    /**
     * Partitions takes the number of partitions as parameter.  Method checks the correct tags after training the model with 
     * equally sized partitions created by line number where each partition gets every nth entry. Partition by random assignment 
     * one of n partitioned groups to be in.  This method appears to have slightly less success (~34500 correct tags for
     * the random and 34974 correct for the line number) than the direct test case in the main assignment
     *
     * @param numPartitions integer number of partitions
     * @param randomPartition
     */
    private static void partitionedTest(int numPartitions, boolean randomPartition) throws Exception {
        Map<String, Map<String, Integer>> initialObsProbMap = new HashMap<>();
        Map<String, Map<String, Integer>> initialTransProbMap = new HashMap<>();
        Map<String, String>[] partitions = new Map[numPartitions];
        for (int i = 0; i < numPartitions; i++) {
            partitions[i] = new HashMap<>();
        }

        // open test readers in array
        BufferedReader[] test = new BufferedReader[]{new BufferedReader(new FileReader("trainingTexts/brown-test-tags.txt")), new BufferedReader(new FileReader("trainingTexts/brown-test-sentences.txt"))};

        String testLine, testCheck;
        int correctTags = 0;
        int checkedTags = 0;
        int lineNum = 0;
        // read file and fill in partition map array by line number
        while ((testLine = test[1].readLine()) != null && (testCheck = test[0].readLine()) != null) {
            if (randomPartition) {
                partitions[(int)(Math.random()*5)].put(testLine, testCheck);      // random partitions
            } else {
                partitions[lineNum % numPartitions].put(testLine, testCheck);       // by line number
            }
            lineNum++;
        }
        // close the readers
        for (BufferedReader bufferedReader : test) {
            bufferedReader.close();
        }

        // train model
        for (int i = 0; i < partitions.length; i++) {
            for (int j = 0; j < partitions.length; j++) {
                if (j != i) {
                    for (String sentence : partitions[j].keySet()) {
                        String[] words = processString(sentence);
                        String[] tags = processString(partitions[j].get(sentence));
                        for (int k = 0; k < words.length; k++) {
                            updateMaps(initialObsProbMap, tags[k], words[k]);
                            if (k == 0) {
                                updateMaps(initialTransProbMap, "#", tags[k]);
                            } else {
                                updateMaps(initialTransProbMap, tags[k - 1], tags[k]);
                            }
                        }
                    }
                }
            }
            Map<String, Map<String, Double>> obsProbMap = createLogProbMap(initialObsProbMap);
            Map<String, Map<String, Double>> transProbMap = createLogProbMap(initialTransProbMap);
            for (String sentence : partitions[i].keySet()) {
                String[] viterbiResult = viterbiPOS(sentence, obsProbMap, transProbMap);
                String[] answers = processString(partitions[i].get(sentence));
                if (viterbiResult.length != answers.length) {
                    System.out.println("Resulting tags are not equal in length");
                } else {
                    for (int k = 0; k < viterbiResult.length; k++) {
                        if (viterbiResult[k].equals(answers[k])) correctTags += 1;
                    }
                    checkedTags += viterbiResult.length;
                }
            }
        }
        System.out.println(correctTags + "/" + checkedTags + " are correct.");
    }
}