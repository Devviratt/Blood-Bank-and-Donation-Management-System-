from django.shortcuts import render,redirect,HttpResponse
from django.contrib import messages
from django.contrib.auth import  logout,login,authenticate
from django.contrib.auth.decorators import login_required
from bbdmsapp.models import CustomUser,Bloodgroup,DonorReg,BloodRequest
from django.contrib.auth import get_user_model
from django.core.exceptions import ObjectDoesNotExist
User = get_user_model()


def DONORSIGNUP(request):
    bg = Bloodgroup.objects.all()
    if request.method == "POST":
        pic = request.FILES.get('pic')
        first_name = request.POST.get('first_name')
        last_name = request.POST.get('last_name')
        username = request.POST.get('username')
        email = request.POST.get('email')
        mobno = request.POST.get('mobno')
        age = request.POST.get('age')
        bg_id = request.POST.get('bg_id')
        gender = request.POST.get('gender')
        address = request.POST.get('address')
        password = request.POST.get('password')

        if CustomUser.objects.filter(email=email).exists():
            messages.warning(request, 'Email already exists')
            return redirect('donorsignup')
        if CustomUser.objects.filter(username=username).exists():
            messages.warning(request, 'Username already exists')
            return redirect('donorsignup')
        else:
            user = CustomUser(
                first_name=first_name,
                last_name=last_name,
                username=username,
                email=email,
                user_type=2,
                profile_pic=pic,
            )
            user.set_password(password)
            user.save()

            bg_instance = Bloodgroup.objects.get(id=bg_id)

            blooddonor = DonorReg(
                admin=user,
                age=age,
                mobilenumber=mobno,
                bloodgroup=bg_instance,
                gender=gender,
                address=address,
            )
            blooddonor.save()

            messages.success(request, 'Signup Successfully')
            return redirect('donorsignup')

    context = {
        'bg': bg
    }

    return render(request, 'donor/donor-signup.html', context)



@login_required(login_url='/')
def BLOODREQUESTDETAILS(request):
    donor_admin = request.user
    donor_id = DonorReg.objects.get(admin=donor_admin)
    bloodreq = BloodRequest.objects.filter(donid=donor_id)
    context = {'bloodreq':bloodreq,

    }
    return render(request,'donor/view-request.html',context)

login_required(login_url='/')
def DONORPROFILE(request):
    bg = Bloodgroup.objects.all()
    donor = DonorReg.objects.get(admin = request.user.id)
    context = {
        "donor":donor,
        "bg":bg,
    }
    return render(request,'donor/donor-profile.html',context)

@login_required(login_url='/')
def DONOR_PROFILE_UPDATE(request):
    if request.method == "POST":
        profile_pic = request.FILES.get('profile_pic')
        first_name = request.POST.get('first_name')
        last_name = request.POST.get('last_name')
        age = request.POST.get('age')
        mobilenumber = request.POST.get('mobilenumber')
        bloodgroup_id = request.POST.get('bloodgroup')
        gender = request.POST.get('gender')
        address = request.POST.get('address')

        try:
            customuser = CustomUser.objects.get(id=request.user.id)
            donor = DonorReg.objects.get(admin=request.user.id)
            bloodgroup = Bloodgroup.objects.get(id=bloodgroup_id)
            
            customuser.first_name = first_name
            customuser.last_name = last_name
            donor.age = age
            donor.mobilenumber = mobilenumber
            donor.bloodgroup = bloodgroup
            donor.gender = gender
            donor.address = address
            
            if profile_pic:
                customuser.profile_pic = profile_pic
                
            customuser.save()
            donor.save()
            
            messages.success(request, "Your profile has been updated successfully")
            return redirect('donor-profile')

        except ObjectDoesNotExist:
            messages.error(request, "Profile update failed: User or Donor not found.")
        except Exception as e:
            messages.error(request, f"An error occurred: {e}")

    return render(request, 'donor/donor-profile.html')