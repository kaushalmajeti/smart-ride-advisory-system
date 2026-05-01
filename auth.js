const COGNITO_CONFIG = {
  UserPoolId: "us-east-1_j9tothbvx",
  ClientId: "6s7pshj5gfatcfls1dkksc4m3g"
};

const userPool = new AmazonCognitoIdentity.CognitoUserPool(COGNITO_CONFIG);

// SIGN UP
function signUp() {
  const email = document.getElementById("email").value.trim();
  const password = document.getElementById("password").value.trim();

  if (!email || !password) {
    showMsg("Please enter email and password.", "error");
    return;
  }

  const attributeList = [
    new AmazonCognitoIdentity.CognitoUserAttribute({ Name: "email", Value: email })
  ];

  userPool.signUp(email, password, attributeList, null, (err, result) => {
    if (err) {
      showMsg(err.message, "error");
      return;
    }
    showMsg("Signup successful! Check your email for the OTP.", "success");
    document.getElementById("otp-section").style.display = "block";
  });
}

// LOGIN
function login() {
  const email = document.getElementById("email").value.trim();
  const password = document.getElementById("password").value.trim();

  if (!email || !password) {
    showMsg("Please enter email and password.", "error");
    return;
  }

  const cognitoUser = new AmazonCognitoIdentity.CognitoUser({
    Username: email,
    Pool: userPool
  });

  // Force USER_PASSWORD_AUTH — bypasses SRP UUID substitution bug
  cognitoUser.setAuthenticationFlowType("USER_PASSWORD_AUTH");

  const authenticationDetails = new AmazonCognitoIdentity.AuthenticationDetails({
    Username: email,
    Password: password
  });

  cognitoUser.authenticateUser(authenticationDetails, {
    onSuccess: function (result) {
      showMsg("Login successful! Redirecting...", "success");
      localStorage.setItem("user", email);
      localStorage.setItem("idToken", result.getIdToken().getJwtToken());
      setTimeout(() => { window.location.href = "index.html"; }, 800);
    },
    onFailure: function (err) {
      console.error("Auth error:", err.code, err.message);
      if (err.code === "UserNotConfirmedException") {
        showMsg("Account not verified. Enter OTP below.", "error");
        document.getElementById("otp-section").style.display = "block";
      } else if (err.code === "NotAuthorizedException") {
        showMsg("Incorrect email or password.", "error");
      } else if (err.code === "UserNotFoundException") {
        showMsg("No account found with this email.", "error");
      } else {
        showMsg(err.message, "error");
      }
    }
  });
}

// OTP VERIFY
function confirmUser() {
  const email = document.getElementById("email").value.trim();
  const otp = document.getElementById("otp").value.trim();

  if (!email || !otp) {
    showMsg("Please enter your email and OTP.", "error");
    return;
  }

  const cognitoUser = new AmazonCognitoIdentity.CognitoUser({
    Username: email,
    Pool: userPool
  });

  cognitoUser.confirmRegistration(otp, true, function (err, result) {
    if (err) {
      showMsg(err.message, "error");
      return;
    }
    showMsg("Account verified! You can now log in.", "success");
    document.getElementById("otp-section").style.display = "none";
  });
}

// HELPER
function showMsg(text, type) {
  const el = document.getElementById("msg");
  el.innerText = text;
  el.className = type; // "success" or "error"
}

function demoLogin() {
  localStorage.setItem("user", "demo@example.com");
  localStorage.setItem("idToken", "demo-session");
  window.location.href = "index.html";
}
