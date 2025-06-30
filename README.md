![image](https://github.com/user-attachments/assets/37d717b3-9d0d-4d57-bed3-b97e3f0d2f98)

주제
## “상식 수준의 의학정보 vs 최신 의학 논문 기반 정보 비교” 를 자동으로 해주는 도구

일반 상식과 최신 논문의 의학정보를 비교함으로 최신 일반 상식의 과학적 오류를 알아 볼 수 있는 챗 봇 시스템입니다.

기존 gpt 4.1 모델에 의학 논문데이터를 보강했습니다.

### 1. 데이터 수집
- PMC , Europe PMC API 크롤링 `건강 상식` 키워드 논문 크롤링
- medirxiv `medicine` 키워드 논문 크롤링
- VectorDB 저장 (ChromaDB)
- AI Model : gpt-4.1
- Embedding Model : text-embedding-3-large

![image](https://github.com/user-attachments/assets/8361f1dd-c0b5-4cd2-ba96-27e5fe9b9714)

![image](https://github.com/user-attachments/assets/405fa90a-910e-48d3-8009-a5736d553923)
