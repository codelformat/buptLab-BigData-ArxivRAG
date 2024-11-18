from django.shortcuts import render
import openai
from rest_framework.decorators import api_view
from django.http import JsonResponse, HttpResponse, HttpRequest


openai.api_key = "REDACTED"


# Create your views here.
def OpenAI():
    return HttpResponse("hello")


@api_view(["Post"])
def searchQuery(request):
    print("hello")
    try:
        query = request.data.get("query", "")
        if not query:
            return JsonResponse({"error": "Query is required"}, status=400)
        try:
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a knowledge base."},
                    {"role": "user", "content": query},
                ],
                max_tokens=200,
                temperature=0.7,
            )
            print("1")
            answer = response.choices[0].message.content
            print(answer)
            return JsonResponse({"response": answer}, status=200)
        except Exception as e:
            import traceback
            print(traceback.format_exc())  # 打印完整的错误堆栈
            return JsonResponse({"error": str(e)}, status=500)

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)
