from django import forms
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm, PasswordChangeForm as DjangoPasswordChangeForm
from django.contrib.auth.models import User
from .models import Profile

class CustomUserCreationForm(UserCreationForm):
    username = forms.CharField(
        max_length=150,
        required=True,
        label='ID',
        help_text='영문, 숫자, @/./+/-/_ 만 사용 가능합니다. (최대 150자)'
    )
    email = forms.EmailField(
        required=True,
        label='이메일 주소',
        help_text='비밀번호 재설정 등에 사용됩니다.'
    )
    password1 = forms.CharField(
        label='비밀번호',
        widget=forms.PasswordInput,
        help_text='8자 이상, 영문/숫자/특수문자 조합 권장'
    )
    password2 = forms.CharField(
        label='비밀번호 확인',
        widget=forms.PasswordInput,
        help_text='동일한 비밀번호를 한 번 더 입력해 주세요.'
    )
    nickname = forms.CharField(max_length=30, required=True, label='별명', help_text='서비스 내에서 사용할 별명(닉네임)을 입력하세요.')

    class Meta(UserCreationForm.Meta):
        fields = ('username', 'email', 'password1', 'password2', 'nickname')

    def save(self, commit=True):
        user = super().save(commit)
        nickname = self.cleaned_data['nickname']
        Profile.objects.create(user=user, nickname=nickname)
        return user

class CustomAuthenticationForm(AuthenticationForm):
    username = forms.CharField(label='ID')
    password = forms.CharField(label='비밀번호', widget=forms.PasswordInput)

class ProfileEditForm(forms.ModelForm):
    nickname = forms.CharField(max_length=30, required=True, label='별명')
    email = forms.EmailField(label='이메일 주소', required=True)
    class Meta:
        model = Profile
        fields = ['nickname']

    def __init__(self, *args, **kwargs):
        user = kwargs.pop('user', None)
        super().__init__(*args, **kwargs)
        if user:
            self.fields['email'].initial = user.email

    def save(self, commit=True, user=None):
        profile = super().save(commit)
        if user:
            user.email = self.cleaned_data['email']
            user.save()
        return profile

class PasswordChangeForm(DjangoPasswordChangeForm):
    old_password = forms.CharField(label='현재 비밀번호', widget=forms.PasswordInput)
    new_password1 = forms.CharField(label='새 비밀번호', widget=forms.PasswordInput)
    new_password2 = forms.CharField(label='새 비밀번호 확인', widget=forms.PasswordInput)

class FindUsernameForm(forms.Form):
    email = forms.EmailField(label='이메일 주소', required=True)

class FindPasswordForm(forms.Form):
    email = forms.EmailField(label='이메일 주소', required=True)
