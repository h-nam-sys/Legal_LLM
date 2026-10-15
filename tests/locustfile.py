from locust import HttpUser, task, between
import random
import uuid

class LegalAssistantUser(HttpUser):
    # Wait between 1 to 5 seconds between tasks to simulate real human typing
    wait_time = between(1, 5)

    def on_start(self):
        """Runs exactly once per simulated user when they spawn."""
        # Generate a fake unique user ID and IP address to bypass SlowAPI limits
        self.user_id = str(uuid.uuid4())
        self.fake_ip = f"{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}"

        self.headers = {
            "x-user-id": self.user_id,
            "X-Forwarded-For": self.fake_ip, # Spoof IP for rate limiter
            "Content-Type": "application/json"
        }

        # 1. Start a new conversation
        response = self.client.post(
            "/start-conversation",
            json={"title": "Stress Test Concurrency"},
            headers=self.headers
        )

        if response.status_code == 200:
            self.conversation_id = response.json().get("conversation_id")
        else:
            self.conversation_id = None

    @task(3) # Weight of 3 means this runs 3x more often than other tasks
    def chat_with_bot(self):
        """Simulates sending a message to the RAG/LLM endpoint."""
        if not self.conversation_id:
            return

        payload = {
            "user_prompt": "Thủ tục đăng ký khai sinh cần những giấy tờ gì?"
        }

        # We hit the chat endpoint. In a real test, this tests Qdrant RAG + LLM Queue
        with self.client.post(
            f"/chat/{self.conversation_id}",
            json=payload,
            headers=self.headers,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            elif response.status_code == 429:
                response.failure("Rate Limited (IP Spoofing Failed)")
            else:
                response.failure(f"Failed with {response.status_code}")

    @task(1)
    def view_history(self):
        """Simulates the user opening their chat history panel."""
        if self.conversation_id:
            self.client.get(f"/history/{self.conversation_id}", headers=self.headers)

    @task(1)
    def view_admin_stats(self):
        """Simulates an admin checking the dashboard (heavy DB count query)."""
        self.client.get("/admin/stats", headers=self.headers)
