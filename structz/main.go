package structz

type Document struct {
	Id     int
	Length int
	Tf     map[string]int
}

type InvertedIndex struct {
	CorpusSize int
	Df         map[string]int
	Idf        map[string]float64
	Terms      map[string][]*Document
}

type SearchResult struct {
	Query     []string
	Documents []*Document
}
