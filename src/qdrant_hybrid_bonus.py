"""Optional Qdrant native dense + sparse hybrid retrieval bonus.

Install the optional dependency first:
    pip install 'qdrant-client[fastembed]'
"""
from qdrant_client import QdrantClient, models


def run_qdrant_hybrid_bonus(chunks, embeddings, questions, model):
    if not chunks:
        raise ValueError("No chunks available for Qdrant indexing.")

    client = QdrantClient(":memory:")
    collection = "week7_day2_hybrid"
    vector_size = int(embeddings.shape[1])

    client.create_collection(
        collection_name=collection,
        vectors_config={
            "dense": models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE
            )
        },
        sparse_vectors_config={
            "bm25": models.SparseVectorParams(
                modifier=models.Modifier.IDF
            )
        }
    )

    points = []
    for index, chunk in enumerate(chunks):
        points.append(
            models.PointStruct(
                id=index,
                vector={
                    "dense": embeddings[index].tolist(),
                    "bm25": models.Document(
                        text=chunk["text"],
                        model="Qdrant/bm25"
                    )
                },
                payload={
                    "chunk_id": chunk["chunk_id"],
                    "text": chunk["text"],
                    "source": chunk["source"]
                }
            )
        )

    client.upload_points(collection_name=collection, points=points)
    print("\nQDRANT BONUS: Native dense + sparse hybrid retrieval")

    for item in questions:
        # Dense vector comes from the same sentence-transformer model used by the project.
        query_vector = model.encode(
            [item["question"]], convert_to_numpy=True,
            normalize_embeddings=True
        )[0].tolist()

        result = client.query_points(
            collection_name=collection,
            prefetch=[
                models.Prefetch(
                    query=query_vector,
                    using="dense",
                    limit=10
                ),
                models.Prefetch(
                    query=models.Document(
                        text=item["question"],
                        model="Qdrant/bm25"
                    ),
                    using="bm25",
                    limit=10
                )
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=3,
            with_payload=True
        )
        found = [p.payload.get("chunk_id", "") for p in result.points]
        print(f"{item['id']} | {item['question']} | {found}")

    client.close()
