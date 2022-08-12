package main

import (
	"errors"
	"fmt"
	"math"
	"nlp/score"
	"nlp/stopwords"
	"nlp/structz"
	"strings"
)

func Tokenizer(document string, normalize bool) []string {
	fields := strings.Fields(document)

	if normalize {

		localSlice := make([]string, len(fields))
		for i, token := range fields {
			localSlice[i] = strings.ToLower(token)
		}

		return localSlice
	}

	return fields

}

func StopWordLang(lang string) (map[string]bool, error) {

	if lang == "en" {
		return stopwords.English, nil
	}

	return nil, errors.New("Not found lang from stopwords")
}

func RemoveStopWords(tokens []string, lang string) ([]string, error) {

	stopWordLang, err := StopWordLang(lang)

	if err != nil {
		return nil, err
	}

	if len(tokens) == 0 {
		return make([]string, 0), nil
	}

	var localSlice = make([]string, 0)

	for _, token := range tokens {
		if !stopWordLang[token] {
			localSlice = append(localSlice, token)
		}
	}

	return localSlice, nil

}

/**
DF - Document Frequency per term.
i.e. Number of documents in the corpus that contains the term.
**/
func DocumentFrequencyPerTerm(term string, documentsTermFrequency []map[string]int) int {

	var count = 0

	for _, localMap := range documentsTermFrequency {
		if localMap[term] != 0 {
			count += 1
		}
	}

	return count

}

/**
TF - Term Frequence
*/
func TermFrequency(tokens []string) map[string]int {

	localMap := make(map[string]int)

	for _, token := range tokens {

		if localMap[token] == 0 {
			localMap[token] = 1
		} else {
			localMap[token] = localMap[token] + 1
		}
	}

	return localMap

}

func MakeInvertedIndex(corpus []string) structz.InvertedIndex {

	index := structz.InvertedIndex{
		Df:         map[string]int{},
		Terms:      make(map[string][]*structz.Document, 0),
		CorpusSize: 0,
	}

	for i, document := range corpus {
		tokens := Tokenizer(document, true)
		tokensNormalized, _ := RemoveStopWords(tokens, "en")

		index.CorpusSize += 1

		doc := structz.Document{
			Id:     i,
			Length: len(tokensNormalized),
			Tf:     TermFrequency(tokensNormalized),
		}

		for _, term := range tokensNormalized {

			if index.Terms[term] != nil {
				documentList := index.Terms[term]
				if notContains(&doc, documentList) {
					index.Terms[term] = append(documentList, &doc)
					index.Df[term] += 1
				}
			} else {
				index.Terms[term] = []*structz.Document{&doc}
				index.Df[term] = 1
			}
		}

	}

	index.Idf = CalcIdf(index.Df, len(corpus))

	return index

}

func CalcIdf(df map[string]int, corpusSize int) map[string]float64 {

	idf := make(map[string]float64)

	for term, frequency := range df {
		//idf[term] = math.log(1 + (corpus_size - freq + 0.5) / (freq + 0.5))
		freq := float64(frequency) + 0.5
		corpusSize := float64(corpusSize)
		idf[term] = math.Log(1 + (corpusSize-freq)/freq)
	}

	return idf

}

func notContains(document *structz.Document, documents []*structz.Document) bool {

	for _, doc := range documents {
		if doc.Id == document.Id {
			return false
		}
	}

	return true
}

func Search(query string, index structz.InvertedIndex) structz.SearchResult {

	documentFound := make(map[int]*structz.Document)
	localQuery, _ := RemoveStopWords(Tokenizer(query, true), "en")

	for _, term := range localQuery {

		if index.Terms[term] != nil {
			documents := index.Terms[term]

			for _, doc := range documents {
				documentFound[doc.Id] = doc
			}
		}
	}

	searchResults := structz.SearchResult{
		Query:     localQuery,
		Documents: make([]*structz.Document, 0),
	}

	for _, doc := range documentFound {
		searchResults.Documents = append(searchResults.Documents, doc)
	}

	return searchResults

	//k1=1.5, b=0.75

}

func main() {

	corpus := []string{
		"Human machine interface for lab abc computer applications",
		"A survey of user opinion of computer system response time",
		"The EPS user interface management system",
		"System and human system engineering testing of EPS",
		"Relation of user perceived response time to error measurement",
		"The generation of random binary unordered trees",
		"The intersection graph of paths in trees",
		"Graph minors IV Widths of trees and well quasi ordering",
		"Graph minors A survey",
	}

	query := "A"
	index := MakeInvertedIndex(corpus)
	searchResult := Search(query, index)

	for _, doc := range searchResult.Documents {

		score := score.BM25(searchResult.Query, doc, index.Idf, index.CorpusSize, 0.75, 1.5)

		fmt.Printf("", score)
		println("Score: ", score, " Doc: ", corpus[doc.Id])

	}

	a := []float64{2, 1, 0}
	b := []float64{1, 0, 1}
	cos, _ := score.CosineSimilarity(a, b)

	fmt.Printf("", cos)

}
