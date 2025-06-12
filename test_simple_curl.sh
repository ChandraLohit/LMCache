#!/bin/bash

# LMCache Prefix Caching Test - Large Context Prompts
# This script tests prefix caching with 600+ token prompts to trigger KV cache reuse

set -e

SERVER_URL="http://localhost:8000"

echo "Testing LMCache vLLM server with large contexts at $SERVER_URL"
echo "================================================================="

# Test 1: Large software architecture context (600+ tokens)
echo "Test 1: Large Software Architecture Context (600+ tokens)"
curl -s -X POST "$SERVER_URL/v1/chat/completions" \
-H "Content-Type: application/json" \
-d @- << 'EOF'
{
  "model": "mistral-7b-instruct",
  "messages": [
    {
      "role": "system", 
      "content": "You are a senior software architect with 15+ years of experience designing large-scale distributed systems, microservices architectures, cloud-native applications, and high-performance computing solutions. You have deep expertise in system design, scalability, reliability, security, and performance optimization. You provide detailed technical guidance with practical implementation strategies, architectural patterns, and best practices for enterprise-level software systems."
    },
    {
      "role": "user", 
      "content": "I need to design a comprehensive e-commerce platform that can handle 1 million concurrent users during Black Friday sales events. The system must support real-time inventory management, payment processing, recommendation engines, fraud detection, order fulfillment, customer service chat, analytics dashboard, mobile applications, third-party integrations, multi-region deployment, disaster recovery, auto-scaling, performance monitoring, security compliance, and seamless user experience across web and mobile interfaces. The platform should be built using modern cloud-native technologies, containerized microservices, event-driven architecture, and distributed data management patterns. Please provide a detailed architectural blueprint including technology stack recommendations, data flow diagrams, scalability strategies, security measures, monitoring approaches, deployment pipelines, and operational procedures for maintaining 99.99% uptime during peak traffic scenarios."
    }
  ],
  "max_tokens": 800,
  "temperature": 0.3
}
EOF

echo -e "\n\n=== REPEATING SAME REQUEST FOR PREFIX CACHING TEST ===\n"

# Test 2: EXACT same request to test prefix caching
echo "Test 2: Identical Request (should hit prefix cache)"
curl -s -X POST "$SERVER_URL/v1/chat/completions" \
-H "Content-Type: application/json" \
-d @- << 'EOF'
{
  "model": "mistral-7b-instruct",
  "messages": [
    {
      "role": "system", 
      "content": "You are a senior software architect with 15+ years of experience designing large-scale distributed systems, microservices architectures, cloud-native applications, and high-performance computing solutions. You have deep expertise in system design, scalability, reliability, security, and performance optimization. You provide detailed technical guidance with practical implementation strategies, architectural patterns, and best practices for enterprise-level software systems."
    },
    {
      "role": "user", 
      "content": "I need to design a comprehensive e-commerce platform that can handle 1 million concurrent users during Black Friday sales events. The system must support real-time inventory management, payment processing, recommendation engines, fraud detection, order fulfillment, customer service chat, analytics dashboard, mobile applications, third-party integrations, multi-region deployment, disaster recovery, auto-scaling, performance monitoring, security compliance, and seamless user experience across web and mobile interfaces. The platform should be built using modern cloud-native technologies, containerized microservices, event-driven architecture, and distributed data management patterns. Please provide a detailed architectural blueprint including technology stack recommendations, data flow diagrams, scalability strategies, security measures, monitoring approaches, deployment pipelines, and operational procedures for maintaining 99.99% uptime during peak traffic scenarios."
    }
  ],
  "max_tokens": 800,
  "temperature": 0.3
}
EOF

echo -e "\n\n"

# Test 3: Machine Learning context with extension (partial prefix match)
echo "Test 3: ML Context with Extension (partial prefix cache)"
curl -s -X POST "$SERVER_URL/v1/chat/completions" \
-H "Content-Type: application/json" \
-d @- << 'EOF'
{
  "model": "mistral-7b-instruct",
  "messages": [
    {
      "role": "system", 
      "content": "You are a machine learning research scientist and data scientist with extensive experience in deep learning, neural networks, computer vision, natural language processing, reinforcement learning, and MLOps. You have published papers in top-tier conferences, led AI research teams, and deployed production ML systems at scale. You provide expert guidance on model architecture design, training optimization, hyperparameter tuning, data preprocessing, feature engineering, model evaluation, deployment strategies, monitoring, and ethical AI considerations."
    },
    {
      "role": "user", 
      "content": "I'm building an autonomous vehicle perception system that needs to perform real-time object detection, semantic segmentation, depth estimation, and trajectory prediction from camera, LiDAR, and radar sensor data. The system must achieve 99.9% accuracy in detecting pedestrians, vehicles, traffic signs, and road markings while processing 30 frames per second with less than 100ms latency. I need recommendations for neural network architectures, training datasets, data augmentation techniques, multi-sensor fusion approaches, edge computing optimization, safety validation methods, regulatory compliance strategies, and continuous learning systems for handling edge cases and improving performance over time. How would you design this comprehensive AI pipeline?"
    }
  ],
  "max_tokens": 600,
  "temperature": 0.2
}
EOF

echo -e "\n\n"

# Test 4: Database optimization context (different domain, new cache)
echo "Test 4: Database Optimization Context (new cache entry)"
curl -s -X POST "$SERVER_URL/v1/chat/completions" \
-H "Content-Type: application/json" \
-d @- << 'EOF'
{
  "model": "mistral-7b-instruct",
  "messages": [
    {
      "role": "system", 
      "content": "You are a database expert and performance engineer with deep knowledge of relational databases, NoSQL systems, distributed databases, data warehousing, query optimization, indexing strategies, replication, sharding, backup and recovery, monitoring, and database administration. You have experience with PostgreSQL, MySQL, MongoDB, Cassandra, Redis, Elasticsearch, ClickHouse, Snowflake, and various cloud database services. You provide expert advice on database design, performance tuning, scalability, reliability, and operational best practices."
    },
    {
      "role": "user", 
      "content": "Our financial trading platform is experiencing severe database performance issues during market open hours when processing 50,000 transactions per second. The system handles stock trades, options, futures, forex, cryptocurrency transactions, real-time market data feeds, risk calculations, compliance reporting, and audit trails. We're using a mix of PostgreSQL for transactional data, Redis for caching, ClickHouse for analytics, and Elasticsearch for log analysis. The main issues are query timeouts, connection pool exhaustion, replication lag, storage I/O bottlenecks, and memory pressure. Please provide a comprehensive database optimization strategy including query tuning, indexing improvements, partitioning schemes, caching layers, connection pooling, read replicas, archiving strategies, monitoring setup, and disaster recovery planning to handle peak trading volumes while maintaining ACID compliance and regulatory requirements."
    }
  ],
  "max_tokens": 700,
  "temperature": 0.1
}
EOF

echo -e "\n\n=== PREFIX CACHING TESTS COMPLETED ===\n"
echo "Check vLLM logs for cache hit/miss information and performance differences!"
echo "First identical requests should show faster processing due to KV cache reuse."