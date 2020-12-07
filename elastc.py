import asyncio
from elasticsearch import AsyncElasticsearch
from elasticsearch.helpers import async_scan

es = AsyncElasticsearch()

async def main():
    async for doc in async_scan(
        client=es,
        query={"query": {"match": {"title": "python"}}},
        index="orders-*"
    ):
        print(doc)

loop = asyncio.get_event_loop()
loop.run_until_complete(main())