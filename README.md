# PDF reader
![](https://github.com/ricard-inho/pdf_reader/blob/main/diagram.png)

## Set up
```
docker compose up -d
docker compose exec pdf_reader bash
streamlit run app.py
```

Add your api keys inside `.env` file with the following IDs
```
OPENAI_API_KEY=
HUGGINGFACEHUB_API_TOKEN=
```
