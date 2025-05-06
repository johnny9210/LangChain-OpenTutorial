from utils.vectordbinterface import DocumentManager
from utils.vectordbinterface import Iterable, Any, Optional, List, Dict
from langchain_core.documents import Document
import faiss
import numpy as np
from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Any, Iterable


class FaissCRUDManager(DocumentManager):
    def __init__(
        self, dim: int = 768, embedding: Optional[Any] = None, **kwargs
    ) -> None:
        """
        FAISS 벡터 데이터베이스 매니저 초기화
        
        Args:
            dim: 임베딩 벡터의 차원
            embedding: 선택적 임베딩 함수
            **kwargs: 추가 인자들
        """
        super().__init__()
        self.dim = dim
        self.embedding = embedding
        
        # FAISS 인덱스 초기화 - IndexIDMap으로 감싸서 ID 지원 추가
        base_index = faiss.IndexFlatL2(dim)  # L2 거리 기반 인덱스 생성
        self.index = faiss.IndexIDMap(base_index)  # ID 매핑 지원 추가
        
        # ID와 메타데이터를 저장할 딕셔너리
        self.document_store = {}
        self.next_id = 0  # 내부 ID 카운터
    
    def upsert(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """
        텍스트를 임베딩하고 FAISS 인덱스에 추가
        
        Args:
            texts: 문서 또는 텍스트
            metadatas: 메타데이터
            ids: 고유 ID, None이면 자동 생성
            **kwargs: 추가 매개변수
        """
        texts_list = list(texts)
        
        if ids is None:
            ids = [str(uuid4()) for _ in range(len(texts_list))]
        
        if metadatas is None:
            metadatas = [{} for _ in range(len(texts_list))]
        
        # 텍스트 임베딩
        if self.embedding:
            embeddings = self.embedding.embed_documents(texts_list)
        else:
            # 임베딩 함수가 없는 경우 처리 (예시로 랜덤 벡터 생성)
            embeddings = [np.random.rand(self.dim).astype('float32') for _ in texts_list]
        
        # FAISS 인덱스에 벡터와 ID 추가
        vectors = np.array(embeddings).astype('float32')
        
        # ID를 정수로 변환 (FAISS는 정수 ID만 지원)
        int_ids = np.array([i + self.next_id for i in range(len(texts_list))], dtype=np.int64)
        
        # 벡터와 ID 함께 추가
        self.index.add_with_ids(vectors, int_ids)
        
        # 메타데이터와 문서 저장
        for i, (text, metadata, user_id) in enumerate(zip(texts_list, metadatas, ids)):
            # 현재 인덱스는 self.next_id + i
            idx = self.next_id + i
            
            self.document_store[user_id] = {
                'index': idx,  # FAISS 인덱스 저장
                'text': text,
                'metadata': metadata
            }
        
        self.next_id += len(texts_list)

    def upsert_parallel(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None,
        batch_size: int = 32,
        workers: int = 10,
        **kwargs: Any,
    ) -> None:
        """
        텍스트를 병렬로 처리하여 임베딩하고 FAISS 인덱스에 추가
        
        Args:
            texts: 문서 또는 텍스트
            metadatas: 메타데이터
            ids: 고유 ID, None이면 자동 생성
            batch_size: 배치 크기
            workers: 작업자 수
            **kwargs: 추가 매개변수
        """
        # 배치 생성
        texts_list = list(texts)
        total = len(texts_list)
        
        if ids is None:
            ids = [str(uuid4()) for _ in range(total)]
            
        if metadatas is None:
            metadatas = [{} for _ in range(total)]
        
        batches = [
            (
                texts_list[i : i + batch_size],
                metadatas[i : i + batch_size] if metadatas else None,
                ids[i : i + batch_size] if ids else None,
            )
            for i in range(0, total, batch_size)
        ]
        
        # 병렬 처리
        with ThreadPoolExecutor(max_workers=workers) as executor:
            list(executor.map(lambda batch: self.upsert(*batch, **kwargs), batches))

    def search(self, query: str, k: int = 10, **kwargs: Any) -> List[Dict]:
        """
        쿼리와 가장 유사한 문서 검색
        
        Args:
            query: 검색 쿼리
            k: 반환할 결과 수
            **kwargs: 필터링 옵션
        
        Returns:
            유사한 문서 리스트
        """
        if not self.document_store:  # 문서가 없는 경우
            return []
            
        # 쿼리 임베딩
        if self.embedding:
            query_embedding = self.embedding.embed_documents([query])[0]
        else:
            # 임베딩 함수가 없는 경우 처리 (예시로 랜덤 벡터 생성)
            query_embedding = np.random.rand(self.dim).astype('float32')
        
        # 벡터를 numpy 배열로 변환
        query_vector = np.array([query_embedding]).astype('float32')
        
        # 검색 수행 - IndexIDMap의 경우 거리와 함께 실제 ID 반환
        distances, indices = self.index.search(query_vector, k)
        
        # 결과 변환
        results = []
        
        # 역방향 인덱스 매핑 (내부 인덱스 ID -> 사용자 ID)
        index_to_id = {}
        for user_id, doc_info in self.document_store.items():
            index_to_id[doc_info['index']] = user_id
        
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx == -1:  # -1은 결과가 없음을 의미
                continue
            
            # 내부 인덱스를 통해 사용자 ID 찾기
            if idx not in index_to_id:
                continue
                
            user_id = index_to_id[idx]
            doc_info = self.document_store[user_id]
            
            # 코사인 유사도로 변환 (선택적)
            score = 1.0 / (1.0 + distance)
            score = round(score, 3)
            
            # 결과 딕셔너리 생성
            result = {
                'text': doc_info['text'],
                'metadata': {
                    'id': user_id,
                    **doc_info['metadata']
                },
                'score': score
            }
            results.append(result)
        
        return results

    def delete(
        self,
        ids: Optional[List[str]] = None,
        filters: Optional[Dict] = None,
        **kwargs: Any,
    ) -> bool:
        """
        인덱스에서 문서 삭제
        
        Args:
            ids: 삭제할 문서의 ID 목록
            filters: 삭제할 문서를 필터링하는 조건
            **kwargs: 추가 매개변수
            
        Returns:
            성공 여부를 나타내는 불리언 값
        """
        # 필터 기반 삭제
        if filters and not ids:
            ids_to_delete = []
            for user_id, doc_info in self.document_store.items():
                match = True
                for key, value in filters.items():
                    if key not in doc_info['metadata'] or doc_info['metadata'][key] != value:
                        match = False
                        break
                
                if match:
                    ids_to_delete.append(user_id)
            
            if ids_to_delete:
                return self.delete(ids=ids_to_delete)
            return True
        
        # ID가 없고 필터도 없으면 모든 문서 삭제
        if ids is None and filters is None:
            # 모든 문서 삭제 - 인덱스를 재설정합니다
            base_index = faiss.IndexFlatL2(self.dim)
            self.index = faiss.IndexIDMap(base_index)
            self.document_store = {}
            self.next_id = 0
            return True
        
        # 특정 ID 삭제
        if ids:
            # 삭제할 ID가 존재하는지 확인
            ids_to_delete = [id for id in ids if id in self.document_store]
            
            if not ids_to_delete:
                return True  # 삭제할 것이 없음
            
            # FAISS 내부 ID 목록 추출
            faiss_ids = [self.document_store[user_id]['index'] for user_id in ids_to_delete]
            
            # IndexIDMap에서 제공하는 remove_ids 메서드 사용
            try:
                self.index.remove_ids(np.array(faiss_ids, dtype=np.int64))
                
                # 문서 저장소에서도 삭제
                for user_id in ids_to_delete:
                    del self.document_store[user_id]
                
                return True
            except Exception as e:
                print(f"FAISS 삭제 오류: {e}")
                return False