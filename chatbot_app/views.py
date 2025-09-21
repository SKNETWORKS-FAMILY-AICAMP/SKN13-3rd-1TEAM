from django.shortcuts import render, redirect
from django.http import JsonResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.decorators import login_required
import json
from .models import ChatMessage, Profile
from .forms import CustomUserCreationForm, CustomAuthenticationForm, ProfileEditForm, PasswordChangeForm, FindUsernameForm, FindPasswordForm
from django.contrib.auth.models import User
from django.utils.crypto import get_random_string
from module import build_compare_graph, CompareState
from dotenv import load_dotenv
import os
from django.contrib import messages
from django.core.mail import send_mail  # 실제 메일 발송용(옵션)

load_dotenv()

rag_graph = build_compare_graph()

def get_rag_response(user_query):
    state = {
        "question": user_query,
        "translated_question": "",
        "social_knowledge": "",
        "latest_docs": [],
        "k_value": 5,
        "final_answer": "",
        "db_has_data": False
    }
    try:
        result = rag_graph.invoke(state)
        return result["final_answer"]
    except Exception as e:
        print(f"RAG Graph 실행 중 오류 발생: {e}")
        return "죄송합니다. 챗봇 응답 생성 중 오류가 발생했습니다."

def get_rag_response_stream(user_query):
    # 실제 LLM 스트림 연동 시에는 chunk 단위로 yield
    answer = get_rag_response(user_query)
    for c in answer:
        yield c

def register_view(request):
    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect('chatbot_view')
    else:
        form = CustomUserCreationForm()
    return render(request, 'chatbot_app/register.html', {'form': form})

def login_view(request):
    if request.method == 'POST':
        form = CustomAuthenticationForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(request, username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect('chatbot_view')
            else:
                form.add_error(None, "Invalid username or password")
    else:
        form = CustomAuthenticationForm()
    return render(request, 'chatbot_app/login.html', {'form': form})

@login_required
def logout_view(request):
    logout(request)
    return redirect('login_view')

@login_required
def profile_edit(request):
    profile, created = Profile.objects.get_or_create(user=request.user)
    if request.method == 'POST':
        form = ProfileEditForm(request.POST, instance=profile, user=request.user)
        if form.is_valid():
            form.save(user=request.user)
            messages.success(request, '회원정보가 성공적으로 변경되었습니다.')
        else:
            messages.error(request, '입력값을 다시 확인해 주세요.')
    else:
        form = ProfileEditForm(instance=profile, user=request.user)
    return render(request, 'chatbot_app/profile_edit.html', {'form': form})

def find_username(request):
    username = None
    if request.method == 'POST':
        form = FindUsernameForm(request.POST)
        if form.is_valid():
            email = form.cleaned_data['email']
            try:
                user = User.objects.get(email=email)
                username = user.username
                messages.success(request, f'해당 이메일로 가입된 아이디는 "{username}" 입니다.')
            except User.DoesNotExist:
                messages.error(request, '해당 이메일로 가입된 계정이 없습니다.')
    else:
        form = FindUsernameForm()
    return render(request, 'chatbot_app/find_username.html', {'form': form, 'username': username})

def find_password(request):
    temp_passwords = []
    if request.method == 'POST':
        form = FindPasswordForm(request.POST)
        if form.is_valid():
            email = form.cleaned_data['email']
            users = User.objects.filter(email=email)
            if users.exists():
                for user in users:
                    temp_password = get_random_string(10)
                    user.set_password(temp_password)
                    user.save()
                    temp_passwords.append((user.username, temp_password))
                msg = "임시 비밀번호가 발급되었습니다:<br>"
                for username, pw in temp_passwords:
                    msg += f'아이디: {username} / 임시 비밀번호: {pw}<br>'
                messages.success(request, msg + "(로그인 후 반드시 비밀번호를 변경하세요)")
            else:
                messages.error(request, '해당 이메일로 가입된 계정이 없습니다.')
    else:
        form = FindPasswordForm()
    return render(request, 'chatbot_app/find_password.html', {'form': form, 'temp_passwords': temp_passwords})

@login_required
def password_change(request):
    password_changed = False
    if request.method == 'POST':
        form = PasswordChangeForm(user=request.user, data=request.POST)
        if form.is_valid():
            form.save()
            password_changed = True
            messages.success(request, '비밀번호가 성공적으로 변경되었습니다. 아래 버튼을 눌러 다시 로그인해 주세요.')
            # redirect하지 않고 바로 렌더링
        else:
            messages.error(request, '입력값을 다시 확인해 주세요.')
    else:
        form = PasswordChangeForm(user=request.user)
    return render(request, 'chatbot_app/password_change.html', {'form': form, 'password_changed': password_changed})

@login_required
def chatbot_view(request):
    chat_messages = ChatMessage.objects.filter(user=request.user).order_by('timestamp')
    nickname = None
    try:
        nickname = request.user.profile.nickname
    except Exception:
        pass
    return render(request, 'chatbot_app/chatbot.html', {'chat_messages': chat_messages, 'nickname': nickname})

@csrf_exempt
@login_required
def chat_api(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            user_message = data.get('message', '')

            bot_response = get_rag_response(user_message)

            ChatMessage.objects.create(user=request.user, user_message=user_message, bot_response=bot_response)

            return JsonResponse({'response': bot_response})
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON'}, status=400)
    return JsonResponse({'error': 'Invalid request method'}, status=405)

@csrf_exempt
@login_required
def chat_api_stream(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            user_message = data.get('message', '')

            def event_stream():
                for chunk in get_rag_response_stream(user_message):
                    yield chunk

            response = StreamingHttpResponse(event_stream(), content_type='text/plain')
            return response
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON'}, status=400)
    return JsonResponse({'error': 'Invalid request method'}, status=405)

@login_required(login_url='login_view')
def home(request):
    nickname = None
    try:
        nickname = request.user.profile.nickname
    except Exception:
        pass
    return render(request, 'chatbot_app/chatbot.html', {'nickname': nickname})
