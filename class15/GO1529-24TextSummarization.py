# GO1529-24TextSummarization
  from sumy.parsers.plaintext import PlaintextParser
  from sumy.summarizers.text_rank import TextRankSummarizer
  parser = PlaintextParser.from_string(texto, Tokenizer("portuguese"))
  summarizer = TextRankSummarizer()
  summary = summarizer(parser.document, 3)
