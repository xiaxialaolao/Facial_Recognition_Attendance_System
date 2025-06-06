<?php
session_start();
include 'config.php';
include 'includes/language-loader.php';
include 'includes/notification-functions.php';
include 'includes/log-functions.php';

// 未登录重定向
if (!isset($_SESSION['loggedin']) || $_SESSION['loggedin'] !== true) {
    header("Location: login.php");
    exit();
}

// 检查是否为管理员
if ($_SESSION['role'] !== 'admin') {
    header("Location: dashboard.php");
    exit();
}

$error = "";
$success = "";

// 处理表单提交
if ($_SERVER["REQUEST_METHOD"] == "POST") {
    // 获取表单数据
    $employee_id = $_POST['employee_id'];
    $username = $_POST['username'];
    $fullname = $_POST['fullname'];
    $role = $_POST['role'];
    $password = $_POST['password'];
    $confirm_password = $_POST['confirm_password'];

    // 验证数据
    if (empty($employee_id) || empty($username) || empty($fullname) || empty($role) || empty($password) || empty($confirm_password)) {
        $error = "All fields are required";
        set_error_notification($error);
    } elseif ($password !== $confirm_password) {
        $error = "Password and confirmation do not match";
        set_error_notification($error);
    } else {
        // 检查员工ID是否已存在
        $check_sql = "SELECT employee_id FROM users WHERE employee_id = ?";
        $check_stmt = $conn->prepare($check_sql);
        $check_stmt->bind_param("i", $employee_id);
        $check_stmt->execute();
        $check_result = $check_stmt->get_result();

        if ($check_result->num_rows > 0) {
            $error = "Employee ID already exists";
        } else {
            // 检查用户名是否已存在
            $check_username_sql = "SELECT username FROM users WHERE username = ?";
            $check_username_stmt = $conn->prepare($check_username_sql);
            $check_username_stmt->bind_param("s", $username);
            $check_username_stmt->execute();
            $check_username_result = $check_username_stmt->get_result();

            if ($check_username_result->num_rows > 0) {
                $error = "Username already exists";
            } else {
                // 开始事务
                $conn->begin_transaction();

                try {
                    // 检查是否有裁剪后的图片数据
                    if (empty($_POST['cropped_image'])) {
                        throw new Exception(__('profile_picture_required'));
                    }

                    // 从base64字符串中提取图片数据
                    $cropped_image = $_POST['cropped_image'];
                    $image_parts = explode(";base64,", $cropped_image);

                    // 确保数据格式正确
                    if (count($image_parts) !== 2) {
                        throw new Exception(__('invalid_image_format'));
                    }

                    $image_data_base64 = $image_parts[1];
                    $image_data = base64_decode($image_data_base64);

                    // 插入新用户
                    // 对密码进行哈希处理
                    $hashed_password = password_hash($password, PASSWORD_DEFAULT);
                    $insert_sql = "INSERT INTO users (employee_id, username, password, fullname, role, profile_picture) VALUES (?, ?, ?, ?, ?, ?)";
                    $insert_stmt = $conn->prepare($insert_sql);
                    $insert_stmt->bind_param("isssss", $employee_id, $username, $hashed_password, $fullname, $role, $image_data);
                    $insert_stmt->execute();

                    // 创建文件名 - 使用用户的全名
                    $filename = $fullname . '.jpg';
                    // 只替换文件名中的特殊字符，保留空格
                    $filename = preg_replace('/[^a-zA-Z0-9_\-\. ]/', '', $filename);
                    $profile_pictures_dir = '/home/xiaxialaolao/FRAS_env/Profile_Pictures';
                    $filepath = $profile_pictures_dir . '/' . $filename;

                    // 确保目录存在
                    if (!file_exists($profile_pictures_dir)) {
                        mkdir($profile_pictures_dir, 0755, true);
                    }

                    // 检查并删除旧的图片文件（如果存在）
                    if (file_exists($filepath)) {
                        unlink($filepath);
                    }

                    // 保存图片到文件系统
                    $result = file_put_contents($filepath, $image_data);

                    if ($result !== false) {
                        // 确保文件权限正确
                        chmod($filepath, 0644);

                        // 强制文件系统同步，确保更改立即可见
                        clearstatcache(true, $filepath);
                    }

                    // 提交事务
                    $conn->commit();

                    // 记录用户添加日志
                    $log_message = "New user added: $username";
                    log_info($log_message, "addition.php");

                    // 设置成功通知
                    set_success_notification(__('user_added'));

                    // 清空表单数据，以便用户可以添加另一个用户
                    $employee_id = "";
                    $username = "";
                    $fullname = "";
                    $role = "user";

                } catch (Exception $e) {
                    // 回滚事务
                    $conn->rollback();
                    $error = $e->getMessage();
                    set_error_notification($error);
                }
            }
        }
    }
}
?>
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="icon" href="icon.png" type="image/x-icon">
    <link rel="stylesheet" href="css/dashboard-style.css">
    <link rel="stylesheet" href="css/slidebar.css">
    <link rel="stylesheet" href="css/addition-style.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/cropperjs/1.5.12/cropper.min.css">
    <?php
    // language-loader.php已在文件顶部包含
    include 'includes/theme-loader.php';
    include 'includes/date-formatter.php';
    include 'includes/notifications-loader.php';
    ?>
    <title><?php echo __('add_new_user'); ?> - FRAS System</title>
</head>
<body>
    <div class="dashboard-container">
        <?php include 'includes/sidebar.php'; ?>

        <!-- 主内容区 -->
        <div class="main-content">
            <!-- 顶部标题区 -->
            <div class="content-header">
                <h2 class="content-title"><?php echo __('Add New User'); ?></h2>

                <button onclick="window.location.href='dashboard.php'" class="action-button">
                    <i class="fas fa-arrow-left"></i>
                    <?php echo __('back_to_dashboard'); ?>
                </button>
            </div>

            <!-- 表单区域 -->
            <div class="form-container" style="max-width: 800px; margin-left: auto; margin-right: auto;">
                <?php if (!empty($error)): ?>
                    <script>
                        document.addEventListener('DOMContentLoaded', function() {
                            showErrorNotification('<?php echo addslashes($error); ?>');
                        });
                    </script>
                <?php endif; ?>

                <?php if (!empty($success)): ?>
                    <script>
                        document.addEventListener('DOMContentLoaded', function() {
                            showSuccessNotification('<?php echo addslashes($success); ?>');
                        });
                    </script>
                <?php endif; ?>

                <form method="POST" enctype="multipart/form-data" onsubmit="return validateForm()">
                    <div class="form-group">
                        <label for="employee_id"><?php echo __('employee_id'); ?>:</label>
                        <input type="number" id="employee_id" name="employee_id" value="<?php echo isset($employee_id) ? htmlspecialchars($employee_id) : ''; ?>" required>
                    </div>

                    <div class="form-group">
                        <label><?php echo __('profile_picture'); ?>: <span class="required"></span></label>
                        <div class="profile-preview">
                            <i class="fas fa-user"></i>
                        </div>
                        <input type="hidden" name="cropped_image" id="cropped_image" required>
                        <label for="profile_picture_input" class="file-input-label">
                            <i class="fas fa-upload"></i> <?php echo __('upload'); ?>
                        </label>
                        <input type="file" id="profile_picture_input" class="file-input" accept="image/*" required>
                        <span id="selected-file-name" style="margin-left: 10px; font-size: 14px;"></span>
                    </div>

                    <div class="form-group">
                        <label for="username"><?php echo __('username'); ?>:</label>
                        <input type="text" id="username" name="username" value="<?php echo isset($username) ? htmlspecialchars($username) : ''; ?>" required>
                    </div>

                    <div class="form-group">
                        <label for="fullname"><?php echo __('full_name'); ?>:</label>
                        <input type="text" id="fullname" name="fullname" value="<?php echo isset($fullname) ? htmlspecialchars($fullname) : ''; ?>" required>
                    </div>

                    <div class="form-group">
                        <label for="role"><?php echo __('role'); ?>:</label>
                        <select id="role" name="role" required>
                            <option value="user" <?php echo (isset($role) && $role == 'user') ? 'selected' : ''; ?>><?php echo __('user'); ?></option>
                            <option value="admin" <?php echo (isset($role) && $role == 'admin') ? 'selected' : ''; ?>><?php echo __('admin'); ?></option>
                        </select>
                    </div>

                    <div class="form-group">
                        <label for="password"><?php echo __('password'); ?>:</label>
                        <input type="password" id="password" name="password" required>
                    </div>

                    <div class="form-group">
                        <label for="confirm_password"><?php echo __('confirm_password'); ?>:</label>
                        <input type="password" id="confirm_password" name="confirm_password" required>
                    </div>

                    <div class="button-group">
                        <a href="dashboard.php" class="cancel-button">
                            <i class="fas fa-times"></i> <?php echo __('cancel'); ?>
                        </a>
                        <button type="submit" class="submit-button">
                            <i class="fas fa-plus"></i> <?php echo __('Add User'); ?>
                        </button>
                    </div>
                </form>
            </div>
        </div>
    </div>

    <!-- Image Cropper Modal -->
    <div class="image-cropper-container" id="image-cropper-modal">
        <div class="cropper-box">
            <div class="cropper-title"><?php echo __('crop_image'); ?></div>
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
            if (!croppedImage) {
                showErrorNotification('<?php echo __('profile_picture_required'); ?>');
                return false;
            }
            return true;
        }

        // Image cropper functionality
        $(document).ready(function() {
            var $imageToCrop = $('#image-to-crop');
            var $cropperModal = $('#image-cropper-modal');
            var $fileInput = $('#profile_picture_input');
            var $croppedImageInput = $('#cropped_image');
            var $selectedFileName = $('#selected-file-name');
            var cropper;

            // When file is selected
            $fileInput.on('change', function(e) {
                var files = e.target.files;
                if (!files || files.length === 0) return;

                var fileName = files[0].name;
                $selectedFileName.text(fileName);

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
                $selectedFileName.text('');
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
                        $('.profile-preview').html('<img src="' + croppedImageData + '" alt="Profile Picture" style="width: 100%; height: 100%; object-fit: cover;">');

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
