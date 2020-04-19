/**
 * Homework 1 - Big Data Computing 19/20
 * Group 15 - Moroldo Luca, Pham Francesco, Occhipinti Michele
 */

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import scala.Tuple2;

import java.io.Serializable;
import java.util.*;
import java.util.stream.Collectors;

public class ClassCount {

    public static void main(String... args) {

        if (args.length != 2) {
            throw new IllegalArgumentException("USAGE: num_partitions file_path");
        }

        SparkConf conf = new SparkConf(true).setAppName("Homework1 - ClassCount");
        JavaSparkContext sc = new JavaSparkContext(conf);
        sc.setLogLevel("WARN");

        int nPartitions = Integer.parseInt(args[0]);
        JavaRDD<String> lines = sc.textFile(args[1]);

        JavaPairRDD<String, Long> genreCount;

        // VERSION WITH DETERMINISTIC PARTITIONS
        genreCount = lines
                .mapToPair((line) -> {
                    // convert lines "id genre" into pairs (k, genre) partitioned in 0..nPartitions-1
                    String[] idGenre = line.split(" ");
                    long id = Long.parseLong(idGenre[0]);
                    String genre = idGenre[1];

                    // map id into range 0..nPartitions-1
                    id = id % nPartitions;
                    return new Tuple2<>(id, genre);
                })
                .groupByKey()
                .flatMapToPair((pair) -> {
                    // count occurrences of each genre within the partition
                    HashMap<String, Long> genreOcc = new HashMap<>();
                    for(String genre : pair._2())
                        genreOcc.put(genre, 1 + genreOcc.getOrDefault(genre, 0L));

                    return getTupleIterator(genreOcc);
                })
                .reduceByKey(Long::sum);


        // collect is required, otherwise keys are sorted locally within each task
        String sPairs = genreCount.sortByKey().collect().stream()
                .map(Tuple2::toString)
                .collect(Collectors.joining(" "));

        System.out.println("VERSION WITH DETERMINISTIC PARTITIONS");
        System.out.println("Output pairs = " + sPairs);

        // VERSION WITH SPARK PARTITIONS
        lines = lines.repartition(nPartitions);

        genreCount = lines
                .map((object) -> {
                    // each input object is a string in the form "id genre"
                    String[] tokens = object.split(" ");
                    return tokens[1]; // discard ids
                })
                .mapPartitionsToPair((partition) -> {
                    long partitionSize = 0; // required to find the biggest partition

                    // count occurrences of each genre within the partition
                    HashMap<String, Long> genreOcc = new HashMap<>();
                    while (partition.hasNext()) {
                        String label = partition.next();
                        genreOcc.put(label, 1L + genreOcc.getOrDefault(label, 0L));
                        partitionSize++;
                    }

                    genreOcc.put("maxPartitionSize", partitionSize); // store current partition size
                    return getTupleIterator(genreOcc);
                });


        System.out.println("VERSION WITH SPARK PARTITIONS");

        // print most frequent genre
        Tuple2<String, Long> maxGenre = genreCount.filter((p) -> !p._1().equals("maxPartitionSize"))
                .reduceByKey(Long::sum)
                .max(new CountComparator());
        System.out.println("Most frequent class = " + maxGenre.toString());

        // print size of biggest partition
        Long maxPartitionSize = genreCount.filter((p) -> p._1().equals("maxPartitionSize"))
                .map(Tuple2::_2)
                .reduce(Math::max);
        System.out.println("Max partition size = " + maxPartitionSize);
    }

    private static Iterator<Tuple2<String, Long>> getTupleIterator(HashMap<String, Long> genreOcc) {
        ArrayList<Tuple2<String, Long>> pairs = new ArrayList<>();
        for (Map.Entry<String, Long> e : genreOcc.entrySet()) {
            pairs.add(new Tuple2<>(e.getKey(), e.getValue()));
        }
        return pairs.iterator();
    }

    static class CountComparator implements Serializable, Comparator<Tuple2<String, Long>> {

        // Returns the pair with higher "long". If the values are equal, then consider the name in alphabetical order
        @Override
        public int compare(Tuple2<String, Long> p1, Tuple2<String, Long> p2) {
            if (p1._2() > p2._2())
                return 1;
            else if (p1._2() < p2._2())
                return -1;
            return p2._1().compareTo(p1._1()); // favor the smaller class in alphabetical order
        }
    }
}
