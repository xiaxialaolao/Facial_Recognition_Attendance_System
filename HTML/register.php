<?php
ini_set('display_errors', 1);
ini_set('display_startup_errors', 1);
error_reporting(E_ALL);

// 设置缓存控制头，防止浏览器缓存
header("Cache-Control: no-store, no-cache, must-revalidate, max-age=0");
header("Cache-Control: post-check=0, pre-check=0", false);
header("Pragma: no-cache");

session_start();
include 'config.php';
include 'includes/log-functions.php';
include 'includes/notification-functions.php';
include 'includes/language-loader.php';

// 如果用户已登录，重定向到仪表盘
if (isset($_SESSION['loggedin']) && $_SESSION['loggedin'] === true) {
    header("Location: dashboard.php");
    exit;
}

// 不再在这里处理表单提交，而是通过AJAX处理
?>
<!DOCTYPE html>
<html lang="UTF-8">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <!-- 防止浏览器缓存 -->
    <meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">
    <meta http-equiv="Pragma" content="no-cache">
    <meta http-equiv="Expires" content="0">
    <link rel="icon" href="icon.png" type="image/x-icon">
    <title><?php echo __('register'); ?> - FRAS System</title>
    <link rel="stylesheet" href="css/login-style.css">
    <link rel="stylesheet" href="css/register-style.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/cropperjs/1.5.12/cropper.min.css">
    <!-- 引入通知系统 -->
    <link rel="stylesheet" href="css/notifications.css">
    <script src="js/notifications.js"></script>
</head>
<body class="register-page">
    <div id="notification-area"></div>

    <div class="login-box register-box" id="register-box">
        <div class="logo-container">
            <img src="icon.png" alt="FRAS Logo" class="logo" id="fras-logo">
        </div>
        <h2 class="title" id="register-title"><?php echo __('register_account'); ?></h2>

        <form method="POST" action="javascript:void(0);" id="register-form" enctype="multipart/form-data" onsubmit="return validateForm()">
            <div class="profile-preview" id="profile-preview">
                <i class="fas fa-user"></i>
            </div>

            <input type="hidden" name="cropped_image" id="cropped_image">
            <label for="profile_picture_input" class="file-input-label">
                <i class="fas fa-upload"></i> <?php echo __('upload_profile_picture'); ?>
            </label>
            <input type="file" id="profile_picture_input" class="file-input" accept="image/*">

            <input type="text" name="employee_id" placeholder="<?php echo __('employee_id'); ?>" required class="input-field">
            <input type="text" name="username" placeholder="<?php echo __('username'); ?>" required class="input-field">
            <input type="text" name="fullname" placeholder="<?php echo __('full_name'); ?>" required class="input-field">
            <input type="password" name="password" placeholder="<?php echo __('password'); ?>" required class="input-field" id="password">
            <input type="password" name="confirm_password" placeholder="<?php echo __('confirm_password'); ?>" required class="input-field" id="confirm_password">

            <button type="submit" class="login-button"><?php echo __('register'); ?></button>
        </form>

        <div class="signup-text">
            <?php echo __('already_have_account'); ?> <a href="login.php"><?php echo __('login_here'); ?></a>
        </div>
    </div>

    <!-- Image Cropper Modal -->
    <div class="image-cropper-container" id="image-cropper-modal">
        <div class="cropper-box">
            <div class="cropper-title"><?php echo __('crop_profile_picture'); ?></div>
            <div class="img-container">
                <img id="image-to-crop" src="" alt="Image to crop">
            </div>
            <div class="img-preview"></div>
            <div class="cropper-buttons">
                <button type="button" class="crop-cancel" id="crop-cancel"><?php echo __('cancel'); ?></button>
                <button type="button" class="crop-submit" id="crop-submit"><?php echo __('apply'); ?></button>
            </div>
        </div>
    </div>

    <!-- JavaScript Libraries -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.6.0/jquery.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/cropperjs/1.5.12/cropper.min.js"></script>

    <script>
        // Form validation function
        function validateForm() {
            var croppedImage = document.getElementById('cropped_image').value;
            var password = document.getElementById('password').value;
            var confirmPassword = document.getElementById('confirm_password').value;

            if (!croppedImage) {
                showErrorNotification('<?php echo __('profile_picture_required'); ?>');
                return false;
            }

            if (password !== confirmPassword) {
                showErrorNotification('<?php echo __('password_mismatch'); ?>');
                return false;
            }

            return true;
        }

        // 处理表单提交
        document.addEventListener('DOMContentLoaded', function() {
            const registerForm = document.getElementById('register-form');

            registerForm.addEventListener('submit', function(e) {
                e.preventDefault();

                // 先验证表单
                if (!validateForm()) {
                    return false;
                }

                // 创建FormData对象
                const formData = new FormData(registerForm);

                // 使用fetch API发送POST请求
                fetch('register-ajax.php', {
                    method: 'POST',
                    body: formData,
                    credentials: 'same-origin'
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        // 注册成功
                        showSuccessNotification(data.message);
                        // 3秒后重定向到登录页面
                        setTimeout(function() {
                            window.location.href = 'login.php';
                        }, 3000);
                    } else {
                        // 注册失败，显示错误
                        showErrorNotification(data.message);
                    }
                })
                .catch(error => {
                    console.error('Error:', error);
                    showErrorNotification('An error occurred. Please try again.');
                });
            });
        });

        // Image cropper functionality
        $(document).ready(function() {
            var $imageToCrop = $('#image-to-crop');
            var $cropperModal = $('#image-cropper-modal');
            var $fileInput = $('#profile_picture_input');
            var $croppedImageInput = $('#cropped_image');
            var $profilePreview = $('#profile-preview');
            var cropper;

            // When file is selected
            $fileInput.on('change', function(e) {
                var files = e.target.files;
                if (!files || files.length === 0) return;

                var reader = new FileReader();
                reader.onload = function(e) {
                    $imageToCrop.attr('src', e.target.result);
                    $cropperModal.css('display', 'block');

                    // Initialize cropper
                    if (cropper) {
                        cropper.destroy();
                    }

                    cropper = new Cropper($imageToCrop[0], {
                        aspectRatio: 1,
                        viewMode: 1,
                        dragMode: 'move',
                        autoCropArea: 1,
                        restore: false,
                        guides: true,
                        center: true,
                        highlight: false,
                        cropBoxMovable: true,
                        cropBoxResizable: true,
                        toggleDragModeOnDblclick: false,
                        preview: '.img-preview'
                    });
                };
                reader.readAsDataURL(files[0]);
            });

            // Cancel cropping
            $('#crop-cancel').on('click', function() {
                if (cropper) {
                    cropper.destroy();
                    cropper = null;
                }
                $cropperModal.css('display', 'none');
                $fileInput.val('');
            });

            // Apply cropping
            $('#crop-submit').on('click', function() {
                if (cropper) {
                    var canvas = cropper.getCroppedCanvas({
                        width: 300,
                        height: 300,
                        minWidth: 100,
                        minHeight: 100,
                        maxWidth: 1000,
                        maxHeight: 1000,
                        fillColor: '#fff',
                        imageSmoothingEnabled: true,
                        imageSmoothingQuality: 'high'
                    });

                    if (canvas) {
                        // Convert canvas to base64 string
                        var croppedImageData = canvas.toDataURL('image/jpeg');
                        $croppedImageInput.val(croppedImageData);

                        // Show preview of cropped image
                        $profilePreview.html('<img src="' + croppedImageData + '" alt="Profile Picture">');

                        // Close modal
                        $cropperModal.css('display', 'none');
                        cropper.destroy();
                        cropper = null;
                    }
                }
            });
        });
    </script>
</body>
</html>
