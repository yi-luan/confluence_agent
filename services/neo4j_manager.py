from neo4j import GraphDatabase

class Neo4jManager:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def add_document_with_entities(self, doc_id: str, entities: list):
        """
        將文檔及其實體加入圖結構
        :param doc_id: 文檔 ID
        :param entities: 文檔中的實體清單
        """
        with self.driver.session() as session:
            session.write_transaction(self._create_document_and_entities, doc_id, entities)

    @staticmethod
    def _create_document_and_entities(tx, doc_id, entities):
        # 創建文檔節點
        tx.run("MERGE (d:Document {id: $doc_id})", doc_id=doc_id)
        for entity in entities:
            # 創建實體節點
            tx.run("MERGE (e:Entity {name: $name, type: $type})", name=entity["text"], type=entity["label"])
            # 創建文檔與實體的關聯
            tx.run("""
                MATCH (d:Document {id: $doc_id}), (e:Entity {name: $name})
                MERGE (d)-[:CONTAINS]->(e)
            """, doc_id=doc_id, name=entity["text"])

    def find_related_documents(self, entity_name: str) -> list:
        """
        查找與某實體相關的文檔
        :param entity_name: 實體名稱
        :return: 相關文檔 ID 清單
        """
        with self.driver.session() as session:
            result = session.read_transaction(self._get_related_documents, entity_name)
        return result

    @staticmethod
    def _get_related_documents(tx, entity_name):
        query = """
        MATCH (e:Entity {name: $entity_name})<-[:CONTAINS]-(d:Document)
        RETURN d.id AS doc_id
        """
        result = tx.run(query, entity_name=entity_name)
        return [record["doc_id"] for record in result]